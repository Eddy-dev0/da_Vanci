"""High fidelity painting pipeline for the robot workflow.

This module orchestrates the full image preparation workflow that is required to
convert an arbitrary RGB input image into structured stroke- and layer
instructions for the painting robot.  It bundles the techniques that were
outlined in the project specification:

* Edge preserving denoising via bilateral/guided filters
* Optional super-resolution (Real-ESRGAN, with fallbacks)
* Local contrast optimisation (CLAHE + sharpening) in the lightness channel
* Colour management in linear space, including device calibration via measured
  paint swatches (3×3 matrix / LUT fitting)
* Palette optimisation that honours the physical paint pots, ΔE2000 matching and
  libimagequant based palette search
* Dithering (Floyd–Steinberg, Jarvis–Judice–Ninke, blue-noise variant)
* Stroke/layer planning using SLIC super-pixels with heuristics for robot
  execution order
* Simulation render and quality metrics (SSIM/LPIPS)
* Simple iterative parameter optimisation to reach a requested similarity

The implementation focusses on clean integration into the existing code base
and keeps every external dependency optional.  Whenever a specialised library is
not available at runtime we gracefully fall back to a robust, albeit simpler,
approach so that the rest of the pipeline still works.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2
from skimage import color as skcolor
from skimage import metrics as skmetrics
from skimage.segmentation import slic

try:  # libimagequant python binding
    import imagequant as liq
except Exception:  # pragma: no cover - optional dependency
    liq = None

try:  # perceptual similarity
    import lpips
except Exception:  # pragma: no cover - optional dependency
    lpips = None

try:
    from colormath.color_objects import LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff import delta_e_cie2000
except Exception:  # pragma: no cover - optional dependency
    LabColor = None
    convert_color = None
    delta_e_cie2000 = None

try:  # Optional guided filter (OpenCV contrib)
    from cv2.ximgproc import guidedFilter as cv_guided_filter
except Exception:  # pragma: no cover - optional dependency
    cv_guided_filter = None

try:  # Super-resolution
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except Exception:  # pragma: no cover - optional dependency
    RealESRGANer = None
    RRDBNet = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from .analyzer import ImageAnalyzer


LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationProfile:
    """Colour calibration profile derived from real-world paint swatches."""

    matrix: np.ndarray
    lut: Optional[np.ndarray] = None
    lut_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(cls) -> "CalibrationProfile":
        return cls(matrix=np.eye(3, dtype=np.float32))

    @classmethod
    def from_swatches(
        cls,
        measured_rgb: np.ndarray,
        painted_rgb: np.ndarray,
        *,
        lut_size: int = 0,
    ) -> "CalibrationProfile":
        """Fit a 3×3 matrix (and optional LUT) from swatch measurements."""

        measured = np.asarray(measured_rgb, dtype=np.float32).reshape(-1, 3)
        painted = np.asarray(painted_rgb, dtype=np.float32).reshape(-1, 3)
        if measured.shape != painted.shape:
            raise ValueError("Swatch arrays must have identical shape")

        if measured.size == 0:
            return cls.identity()

        A = np.hstack([measured, np.ones((measured.shape[0], 1), dtype=np.float32)])
        B = painted
        X, *_ = np.linalg.lstsq(A, B, rcond=None)
        matrix = X[:3, :].T.astype(np.float32)
        offset = X[3:, :].astype(np.float32)

        lut = None
        if lut_size and lut_size >= 2:
            lin = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
            grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1)
            grid_flat = grid.reshape(-1, 3)
            fitted = np.clip(grid_flat @ matrix.T + offset, 0.0, 1.0)
            lut = fitted.reshape(lut_size, lut_size, lut_size, 3)

        profile = cls(matrix=matrix, lut=lut, lut_size=int(lut_size))
        profile.metadata["offset"] = offset
        profile.metadata["swatch_count"] = int(measured.shape[0])
        return profile

    def apply(self, rgb_linear: np.ndarray) -> np.ndarray:
        """Apply the calibration profile to ``rgb_linear`` (float32 0-1)."""

        if rgb_linear.ndim != 3 or rgb_linear.shape[2] != 3:
            raise ValueError("Expected image with shape (H, W, 3)")

        img = np.asarray(rgb_linear, dtype=np.float32)

        if self.lut is not None and self.lut_size >= 2:
            idx = np.clip(img * (self.lut_size - 1), 0, self.lut_size - 1 - 1e-6)
            idx_floor = np.floor(idx).astype(np.int32)
            frac = idx - idx_floor
            # trilinear interpolation
            x0, y0, z0 = idx_floor[..., 0], idx_floor[..., 1], idx_floor[..., 2]
            x1 = np.clip(x0 + 1, 0, self.lut_size - 1)
            y1 = np.clip(y0 + 1, 0, self.lut_size - 1)
            z1 = np.clip(z0 + 1, 0, self.lut_size - 1)

            def lut_at(x, y, z):
                return self.lut[x, y, z]

            c000 = lut_at(x0, y0, z0)
            c100 = lut_at(x1, y0, z0)
            c010 = lut_at(x0, y1, z0)
            c110 = lut_at(x1, y1, z0)
            c001 = lut_at(x0, y0, z1)
            c101 = lut_at(x1, y0, z1)
            c011 = lut_at(x0, y1, z1)
            c111 = lut_at(x1, y1, z1)

            fx, fy, fz = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]

            c00 = c000 * (1 - fx) + c100 * fx
            c01 = c001 * (1 - fx) + c101 * fx
            c10 = c010 * (1 - fx) + c110 * fx
            c11 = c011 * (1 - fx) + c111 * fx
            c0 = c00 * (1 - fy) + c10 * fy
            c1 = c01 * (1 - fy) + c11 * fy
            calibrated = c0 * (1 - fz) + c1 * fz
        else:
            offset = self.metadata.get("offset")
            if offset is None:
                offset = np.zeros((1, 3), dtype=np.float32)
            calibrated = img @ self.matrix.T + offset

        return np.clip(calibrated, 0.0, 1.0)


@dataclass
class PaletteResult:
    palette_rgb: np.ndarray
    indexed_image: np.ndarray
    delta_e_map: np.ndarray


@dataclass
class StrokeInstruction:
    color_rgb: Tuple[int, int, int]
    stage: str
    tool: str
    technique: str
    coverage: float
    path: List[Tuple[int, int]]


@dataclass
class PipelineResult:
    processed_rgb: np.ndarray
    calibrated_rgb: np.ndarray
    palette: PaletteResult
    dithered_rgb: np.ndarray
    stroke_plan: List[StrokeInstruction]
    simulation_rgb: np.ndarray
    metrics: Dict[str, float]
    config: Dict[str, Any]


def _sr_default_model(scale: int) -> Optional[RRDBNet]:  # pragma: no cover - heavy init
    if RRDBNet is None:
        return None
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)


def _linearise_srgb(rgb01: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb01, dtype=np.float32)
    a = 0.055
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)


def _encode_srgb(rgb_linear: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(
        rgb_linear <= 0.0031308,
        rgb_linear * 12.92,
        (1 + a) * np.power(rgb_linear, 1 / 2.4) - a,
    )


class PaintingPipeline:
    """Comprehensive processing pipeline that mirrors the project checklist."""

    def __init__(self) -> None:
        self.analyzer = ImageAnalyzer()
        self._lpips_model = None
        self._realesrgan: Optional[RealESRGANer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        image_source: Any,
        *,
        enable_superres: bool = False,
        superres_scale: int = 2,
        superres_model_path: Optional[str] = None,
        bilateral_diameter: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        guided_radius: int = 8,
        guided_eps: float = 1e-4,
        apply_guided_filter: bool = False,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: int = 8,
        sharpen_amount: float = 0.35,
        calibration_profile: Optional[CalibrationProfile] = None,
        palette_size: int = 12,
        palette_colors: Optional[np.ndarray] = None,
        dither: str = "floyd_steinberg",
        slic_segments: int = 450,
        slic_compactness: float = 18.0,
        stroke_spacing_px: int = 3,
        target_metrics: Optional[Dict[str, float]] = None,
        optimisation_passes: int = 0,
    ) -> PipelineResult:
        """Run the entire workflow on ``image_source``."""

        rgb01 = self.analyzer._ensure_rgb01(image_source)
        rgb_linear = _linearise_srgb(rgb01)

        denoised = self._denoise(rgb_linear, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space,
                                 apply_guided_filter, guided_radius, guided_eps)

        if enable_superres:
            sr = self._run_super_resolution(
                denoised,
                scale=superres_scale,
                model_path=superres_model_path,
            )
        else:
            sr = denoised

        tone_mapped = self._apply_clahe_and_sharpen(sr, clahe_clip_limit, clahe_grid_size, sharpen_amount)

        calibrated = self._apply_calibration(tone_mapped, calibration_profile)

        palette = self._reduce_palette(calibrated, palette_size, palette_colors)

        dithered = self._apply_dithering(calibrated, palette, method=dither)

        stroke_plan = self._plan_strokes(dithered, palette, slic_segments, slic_compactness, stroke_spacing_px)

        simulation = self._render_simulation(palette, dithered.shape[:2])

        metrics = self._evaluate_quality(rgb01, simulation)

        result = PipelineResult(
            processed_rgb=sr,
            calibrated_rgb=calibrated,
            palette=palette,
            dithered_rgb=dithered,
            stroke_plan=stroke_plan,
            simulation_rgb=simulation,
            metrics=metrics,
            config={
                "enable_superres": enable_superres,
                "superres_scale": superres_scale,
                "dither": dither,
                "palette_size": palette_size,
                "slic_segments": slic_segments,
                "slic_compactness": slic_compactness,
                "stroke_spacing_px": stroke_spacing_px,
                "calibration_profile": calibration_profile,
                "palette_colors": palette_colors,
            },
        )

        if optimisation_passes and target_metrics:
            result = self._optimise_parameters(
                image_source,
                base=result,
                target_metrics=target_metrics,
                passes=optimisation_passes,
            )

        return result

    # ------------------------------------------------------------------
    # Individual processing stages
    # ------------------------------------------------------------------
    def _denoise(
        self,
        rgb_linear: np.ndarray,
        diameter: int,
        sigma_color: float,
        sigma_space: float,
        apply_guided_filter: bool,
        guided_radius: int,
        guided_eps: float,
    ) -> np.ndarray:
        rgb = np.clip(rgb_linear, 0.0, 1.0)
        bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr = cv2.bilateralFilter(bgr, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        if apply_guided_filter and cv_guided_filter is not None:
            guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for c in range(3):
                bgr[..., c] = cv_guided_filter(guide, bgr[..., c], guided_radius, guided_eps)
        filtered = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return _linearise_srgb(np.clip(filtered, 0.0, 1.0))

    def _run_super_resolution(
        self,
        rgb_linear: np.ndarray,
        *,
        scale: int,
        model_path: Optional[str],
    ) -> np.ndarray:
        if RealESRGANer is None or RRDBNet is None or torch is None:
            LOGGER.warning("Real-ESRGAN not available – skipping super-resolution.")
            return rgb_linear

        if self._realesrgan is None:
            model = _sr_default_model(scale)
            if model is None:
                LOGGER.warning("RRDBNet arch unavailable – skipping super-resolution.")
                return rgb_linear

            model_path = model_path or os.environ.get("PAINTER_REAL_ESRGAN_MODEL")
            if not model_path or not os.path.exists(model_path):
                LOGGER.warning("Real-ESRGAN model weights missing – skipping super-resolution.")
                return rgb_linear

            self._realesrgan = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                half=torch.cuda.is_available(),
            )

        bgr = cv2.cvtColor((_encode_srgb(np.clip(rgb_linear, 0.0, 1.0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        try:
            output, _ = self._realesrgan.enhance(bgr)
        except RuntimeError as exc:  # pragma: no cover - GPU/weight issues
            LOGGER.warning("Real-ESRGAN inference failed: %s", exc)
            return rgb_linear

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return _linearise_srgb(np.clip(rgb, 0.0, 1.0))

    def _apply_clahe_and_sharpen(
        self,
        rgb_linear: np.ndarray,
        clip_limit: float,
        grid_size: int,
        sharpen_amount: float,
    ) -> np.ndarray:
        rgb = np.clip(rgb_linear, 0.0, 1.0)
        lab = skcolor.rgb2lab(_encode_srgb(rgb))
        L = lab[..., 0] / 100.0
        clahe = cv2.createCLAHE(clipLimit=max(0.1, clip_limit), tileGridSize=(grid_size, grid_size))
        L_clahe = clahe.apply((L * 255).astype(np.uint8)).astype(np.float32) / 255.0
        lab[..., 0] = np.clip(L_clahe * 100.0, 0.0, 100.0)

        sharpen = cv2.GaussianBlur(L_clahe, (0, 0), 1.0)
        enhanced_L = np.clip(L_clahe + sharpen_amount * (L_clahe - sharpen), 0.0, 1.0)
        lab[..., 0] = enhanced_L * 100.0

        rgb_out = skcolor.lab2rgb(lab)
        return _linearise_srgb(np.clip(rgb_out, 0.0, 1.0))

    def _apply_calibration(
        self,
        rgb_linear: np.ndarray,
        profile: Optional[CalibrationProfile],
    ) -> np.ndarray:
        if profile is None:
            return np.clip(rgb_linear, 0.0, 1.0)
        return profile.apply(rgb_linear)

    # ------------------------------------------------------------------
    # Palette + Dithering
    # ------------------------------------------------------------------
    def _reduce_palette(
        self,
        rgb_linear: np.ndarray,
        palette_size: int,
        palette_colors: Optional[np.ndarray],
    ) -> PaletteResult:
        rgb = _encode_srgb(np.clip(rgb_linear, 0.0, 1.0))
        H, W, _ = rgb.shape
        img_uint8 = (rgb * 255).astype(np.uint8)

        if palette_colors is not None:
            palette = np.clip(np.asarray(palette_colors, dtype=np.float32), 0, 1)
            if palette.max() > 1.0:
                palette /= 255.0
        else:
            palette = self._palette_via_libimagequant(img_uint8, palette_size)
            if palette is None:
                palette = self._palette_via_kmeans(rgb_linear, palette_size)

        palette = palette.reshape(-1, 3)
        palette_lab = skcolor.rgb2lab(palette.reshape(-1, 1, 1, 3)).reshape(-1, 3)

        lab = skcolor.rgb2lab(rgb.reshape(H, W, 1, 3)).reshape(H * W, 3)
        use_ciede2000 = delta_e_cie2000 is not None and LabColor is not None and convert_color is not None

        if use_ciede2000:
            palette_lab_objects = [LabColor(*lab_vals) for lab_vals in palette_lab]
            indexed = np.zeros((H * W,), dtype=np.int32)
            delta_e = np.zeros((H * W,), dtype=np.float32)

            for i, lab_px in enumerate(lab):
                ref = LabColor(*lab_px)
                best_idx = 0
                best_de = float("inf")
                for idx, pal_lab in enumerate(palette_lab_objects):
                    de = delta_e_cie2000(ref, pal_lab)
                    if de < best_de:
                        best_de = de
                        best_idx = idx
                indexed[i] = best_idx
                delta_e[i] = best_de
        else:
            diff = palette_lab[None, :, :] - lab[:, None, :]
            dist_sq = np.sum(diff * diff, axis=2)
            indexed = np.argmin(dist_sq, axis=1).astype(np.int32)
            delta_e = np.sqrt(dist_sq[np.arange(dist_sq.shape[0]), indexed]).astype(np.float32)

        indexed_image = indexed.reshape(H, W)
        cleaned_indexed = self._despeckle_index_map(indexed_image)
        if not np.array_equal(cleaned_indexed, indexed_image):
            indexed = cleaned_indexed.reshape(-1)
            if use_ciede2000:
                delta_e = self._compute_delta_e_map(lab, palette_lab_objects, palette_lab, indexed)
            else:
                delta_e = self._compute_delta_e_map(lab, None, palette_lab, indexed)
            indexed_image = cleaned_indexed

        delta_e_map = delta_e.reshape(H, W)
        return PaletteResult(
            palette_rgb=np.clip(palette, 0.0, 1.0),
            indexed_image=indexed_image,
            delta_e_map=delta_e_map,
        )

    def _palette_via_libimagequant(
        self,
        img_uint8: np.ndarray,
        palette_size: int,
    ) -> Optional[np.ndarray]:  # pragma: no cover - depends on optional dep
        if liq is None:
            return None

        H, W, _ = img_uint8.shape
        try:
            quant = liq.ImageQuantizer(max_colors=palette_size)
            quant.set_quality(90, 100)
            result = quant.quantize_pixels(img_uint8)
            palette = np.asarray(result.palette, dtype=np.float32) / 255.0
            if palette.shape[0] == 0:
                return None
            return palette
        except Exception as exc:
            LOGGER.warning("libimagequant failed, falling back to k-means: %s", exc)
            return None

    def _palette_via_kmeans(self, rgb_linear: np.ndarray, palette_size: int) -> np.ndarray:
        from sklearn.cluster import KMeans

        H, W, _ = rgb_linear.shape
        data = rgb_linear.reshape(-1, 3)
        try:
            kmeans = KMeans(n_clusters=palette_size, n_init="auto", random_state=0)
        except TypeError:  # pragma: no cover - older scikit-learn
            kmeans = KMeans(n_clusters=palette_size, n_init=10, random_state=0)
        kmeans.fit(data)
        return np.clip(kmeans.cluster_centers_, 0.0, 1.0)

    def _compute_delta_e_map(
        self,
        lab_pixels: np.ndarray,
        palette_lab_objects: Optional[List[LabColor]],
        palette_lab: np.ndarray,
        assignments: np.ndarray,
    ) -> np.ndarray:
        if palette_lab_objects is not None and delta_e_cie2000 is not None and LabColor is not None:
            delta_e = np.zeros(assignments.shape[0], dtype=np.float32)
            for i, palette_index in enumerate(assignments):
                ref = LabColor(*lab_pixels[i])
                delta_e[i] = float(delta_e_cie2000(ref, palette_lab_objects[int(palette_index)]))
            return delta_e

        diff = palette_lab[assignments] - lab_pixels
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)

    def _despeckle_index_map(self, index_map: np.ndarray, *, min_majority: int = 5) -> np.ndarray:
        """Suppress isolated palette assignments that create single-pixel speckles."""

        if min_majority <= 0:
            return index_map

        height, width = index_map.shape
        padded = np.pad(index_map, 1, mode="edge")
        cleaned = index_map.copy()

        for y in range(height):
            for x in range(width):
                window = padded[y : y + 3, x : x + 3]
                center = window[1, 1]
                values, counts = np.unique(window, return_counts=True)
                majority_idx = int(values[np.argmax(counts)])
                majority_count = int(counts[np.argmax(counts)])
                if majority_idx != center and majority_count >= min_majority:
                    cleaned[y, x] = majority_idx

        return cleaned

    def _apply_dithering(
        self,
        calibrated_rgb: np.ndarray,
        palette: PaletteResult,
        *,
        method: str,
    ) -> np.ndarray:
        rgb = _encode_srgb(np.clip(calibrated_rgb, 0.0, 1.0))
        H, W, _ = rgb.shape
        palette_rgb = palette.palette_rgb

        if method == "blue_noise":
            blue_noise = self._blue_noise_mask((H, W))
            rgb = np.clip(rgb + blue_noise[..., None], 0.0, 1.0)
            method = "floyd_steinberg"

        if method == "jarvis_judice_ninke":
            diffusion = [
                (0, 1, 7 / 48),
                (0, 2, 5 / 48),
                (1, -2, 3 / 48),
                (1, -1, 5 / 48),
                (1, 0, 7 / 48),
                (1, 1, 5 / 48),
                (1, 2, 3 / 48),
                (2, -2, 1 / 48),
                (2, -1, 3 / 48),
                (2, 0, 5 / 48),
                (2, 1, 3 / 48),
                (2, 2, 1 / 48),
            ]
        else:
            diffusion = [
                (0, 1, 7 / 16),
                (1, -1, 3 / 16),
                (1, 0, 5 / 16),
                (1, 1, 1 / 16),
            ]

        out = rgb.copy()
        for y in range(H):
            for x in range(W):
                old_pixel = out[y, x]
                idx = palette.indexed_image[y, x]
                new_pixel = palette_rgb[idx]
                out[y, x] = new_pixel
                quant_error = old_pixel - new_pixel
                for dy, dx, weight in diffusion:
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        out[ny, nx] += quant_error * weight

        return np.clip(out, 0.0, 1.0)

    def _blue_noise_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng(12345)
        noise = rng.standard_normal(shape)
        noise_fft = np.fft.rfftn(noise)
        fy = np.fft.fftfreq(shape[0])[:, None]
        fx = np.fft.rfftfreq(shape[1])[None, :]
        radius = np.sqrt(fx * fx + fy * fy)
        radius[0, 0] = 1.0
        noise_fft *= radius
        blue = np.fft.irfftn(noise_fft, s=shape).real
        blue = (blue - blue.min()) / (blue.max() - blue.min() + 1e-6)
        return (blue - 0.5) * 0.08

    # ------------------------------------------------------------------
    # Stroke planning & simulation
    # ------------------------------------------------------------------
    def _plan_strokes(
        self,
        dithered_rgb: np.ndarray,
        palette: PaletteResult,
        slic_segments: int,
        slic_compactness: float,
        stroke_spacing_px: int,
    ) -> List[StrokeInstruction]:
        rgb = dithered_rgb
        H, W, _ = rgb.shape
        slic_labels = slic(
            rgb,
            n_segments=max(20, slic_segments),
            compactness=max(0.1, slic_compactness),
            start_label=0,
        )

        instructions: List[StrokeInstruction] = []

        for label in np.unique(slic_labels):
            mask = slic_labels == label
            if np.count_nonzero(mask) == 0:
                continue

            palette_indices, counts = np.unique(palette.indexed_image[mask], return_counts=True)
            dominant_idx = int(palette_indices[np.argmax(counts)])
            dominant_color = tuple((palette.palette_rgb[dominant_idx] * 255).astype(np.uint8))

            contours = self.analyzer.mask_to_offset_contours(
                (mask.astype(np.uint8) * 255),
                spacing_px=stroke_spacing_px,
                min_len=8,
            )

            coverage = float(np.count_nonzero(mask)) / float(H * W)
            stage = self._stage_from_coverage(coverage, palette.delta_e_map[mask])
            tool, technique = self._tool_for_stage(stage, coverage)

            for path in contours:
                if len(path) < 2:
                    continue
                instructions.append(
                    StrokeInstruction(
                        color_rgb=dominant_color,
                        stage=stage,
                        tool=tool,
                        technique=technique,
                        coverage=coverage,
                        path=path,
                    )
                )

        instructions.sort(key=lambda inst: (self._stage_priority(inst.stage), -inst.coverage))
        return instructions

    def _stage_from_coverage(self, coverage: float, delta_e_values: np.ndarray) -> str:
        mean_delta = float(np.mean(delta_e_values)) if delta_e_values.size else 0.0
        if coverage > 0.3 and mean_delta < 8:
            return "background"
        if coverage > 0.1:
            return "mid"
        return "detail"

    def _tool_for_stage(self, stage: str, coverage: float) -> Tuple[str, str]:
        if stage == "background":
            return ("wide_brush", "broad_fill")
        if stage == "mid":
            return ("round_brush", "layered_strokes")
        if coverage < 0.05:
            return ("fine_brush", "precision_strokes")
        return ("flat_brush", "edge_refine")

    def _stage_priority(self, stage: str) -> int:
        return {"background": 0, "mid": 1, "detail": 2}.get(stage, 1)

    def _render_simulation(self, palette: PaletteResult, shape: Tuple[int, int]) -> np.ndarray:
        H, W = shape
        rgb = palette.palette_rgb[palette.indexed_image]
        return np.clip(rgb.reshape(H, W, 3), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Quality metrics & optimisation
    # ------------------------------------------------------------------
    def _evaluate_quality(self, reference_rgb01: np.ndarray, simulation_rgb01: np.ndarray) -> Dict[str, float]:
        ref = np.clip(reference_rgb01, 0.0, 1.0)
        sim = np.clip(simulation_rgb01, 0.0, 1.0)
        try:
            ssim_val = skmetrics.structural_similarity(ref, sim, channel_axis=-1, data_range=1.0)
        except TypeError:  # pragma: no cover - older skimage
            ssim_val = skmetrics.structural_similarity(ref, sim, multichannel=True, data_range=1.0)
        metrics: Dict[str, float] = {"ssim": float(ssim_val)}

        if lpips is not None and torch is not None:
            if self._lpips_model is None:
                self._lpips_model = lpips.LPIPS(net="alex")
            ref_tensor = torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
            sim_tensor = torch.from_numpy(sim.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
            with torch.no_grad():  # pragma: no cover - requires torch
                lpips_val = float(self._lpips_model(ref_tensor, sim_tensor).item())
            metrics["lpips"] = lpips_val
        else:
            metrics["lpips"] = float("nan")

        return metrics

    def _optimise_parameters(
        self,
        image_source: Any,
        *,
        base: PipelineResult,
        target_metrics: Dict[str, float],
        passes: int,
    ) -> PipelineResult:
        palette_sizes = [max(4, base.config["palette_size"] - 4), base.config["palette_size"], base.config["palette_size"] + 4]
        dither_methods = [base.config["dither"], "jarvis_judice_ninke", "blue_noise"]

        best = base
        best_score = self._metric_score(base.metrics, target_metrics)

        for _ in range(max(1, passes)):
            improved = False
            for palette_size in palette_sizes:
                for dither_method in dither_methods:
                    candidate = self.process(
                        image_source,
                        enable_superres=base.config["enable_superres"],
                        superres_scale=base.config["superres_scale"],
                        dither=dither_method,
                        palette_size=palette_size,
                        calibration_profile=base.config.get("calibration_profile"),
                        palette_colors=base.config.get("palette_colors"),
                        optimisation_passes=0,
                    )
                    score = self._metric_score(candidate.metrics, target_metrics)
                    if score > best_score:
                        best, best_score = candidate, score
                        improved = True
            if not improved:
                break
        return best

    def _metric_score(self, metrics: Dict[str, float], targets: Dict[str, float]) -> float:
        score = 0.0
        for key, target in targets.items():
            value = metrics.get(key)
            if value is None or np.isnan(value):
                continue
            if key == "lpips":
                score += max(0.0, (target - value))
            else:
                score += max(0.0, value - target)
        return score


__all__ = [
    "CalibrationProfile",
    "PaletteResult",
    "PipelineResult",
    "StrokeInstruction",
    "PaintingPipeline",
]

