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

import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
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

if lpips is not None:  # pragma: no cover - optional dependency
    try:
        import torchvision.models as _tv_models
        from torchvision.models import AlexNet_Weights as _TorchvisionAlexNetWeights
    except Exception:  # pragma: no cover - optional dependency
        _tv_models = None
        _TorchvisionAlexNetWeights = None
    else:
        _orig_alexnet = _tv_models.alexnet

        if not getattr(_orig_alexnet, "_lpips_weights_patch", False):

            def _alexnet_with_weights(*args, **kwargs):
                pretrained = kwargs.pop("pretrained", None)
                weights = kwargs.pop("weights", None)
                if weights is None and pretrained is not None:
                    if pretrained and _TorchvisionAlexNetWeights is not None:
                        weights = _TorchvisionAlexNetWeights.IMAGENET1K_V1
                    else:
                        weights = None
                return _orig_alexnet(*args, weights=weights, **kwargs)

            _alexnet_with_weights._lpips_weights_patch = True  # type: ignore[attr-defined]
            _tv_models.alexnet = _alexnet_with_weights

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

from .analyzer import ImageAnalyzer, _normalise_mask


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
    post_processed_rgb: np.ndarray
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


def enhance_layer(
    layer_rgb01: np.ndarray,
    mask: np.ndarray,
    *,
    scale: int,
    model_path: Optional[str] = None,
) -> Tuple[np.ndarray, bool]:
    """Enhance a single RGB layer via the pipeline's super-resolution stack."""

    rgb = np.clip(np.asarray(layer_rgb01, dtype=np.float32), 0.0, 1.0)
    mask_norm = _normalise_mask(mask)

    if mask_norm is None:
        raise ValueError("mask must not be None")

    pipeline = PaintingPipeline()
    rgb_linear = _linearise_srgb(rgb)
    enhanced_linear, applied = pipeline._run_super_resolution(
        rgb_linear, scale=scale, model_path=model_path
    )

    enhanced_rgb = _encode_srgb(np.clip(enhanced_linear, 0.0, 1.0)).astype(np.float32)

    if mask_norm.shape[:2] != enhanced_rgb.shape[:2]:
        mask_resized = cv2.resize(
            mask_norm,
            (enhanced_rgb.shape[1], enhanced_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
    else:
        mask_resized = mask_norm.astype(np.float32)

    masked = enhanced_rgb * mask_resized[..., None]
    return masked, applied


def _lab_to_srgb_safe(lab: np.ndarray) -> np.ndarray:
    lab_arr = np.asarray(lab, dtype=np.float32)
    xyz = skcolor.lab2xyz(lab_arr)
    xyz[..., 2] = np.clip(xyz[..., 2], 0.0, None)
    return skcolor.xyz2rgb(xyz, clip=True)


def _initialise_lpips_model() -> Optional[Any]:  # pragma: no cover - optional dependency
    if lpips is None or torch is None:
        return None

    try:
        model = lpips.LPIPS(net="alex", pretrained=False, verbose=True)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        LOGGER.warning("Failed to initialise LPIPS model: %s", exc)
        return None

    try:
        weights_root = Path(inspect.getfile(lpips.LPIPS)).resolve().parent / "weights"
        weights_path = weights_root / f"v{model.version}" / f"{model.pnet_type}.pth"
    except Exception as exc:  # pragma: no cover - depends on optional dep
        LOGGER.warning("Failed to resolve LPIPS weights location: %s", exc)
        return None

    if not weights_path.exists():  # pragma: no cover - depends on optional dep
        LOGGER.warning("LPIPS weights not found at %s", weights_path)
        return None

    try:
        try:
            state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - torch < 2.1
            state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        LOGGER.warning("Failed to load LPIPS weights: %s", exc)
        return None

    model.eval()
    return model


class PaintingPipeline:
    """Comprehensive processing pipeline that mirrors the project checklist."""

    def __init__(self) -> None:
        self.analyzer = ImageAnalyzer()
        self._lpips_model = None
        self._realesrgan: Optional[RealESRGANer] = None
        self._superres_supported = (
            RealESRGANer is not None and RRDBNet is not None and torch is not None
        )
        self._superres_warning_emitted = False
        self._superres_weights_warning_emitted = False

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

        exact_palette_mode = bool(
            palette_colors is None and (palette_size is None or palette_size <= 0)
        )

        if exact_palette_mode:
            denoised = np.clip(rgb_linear, 0.0, 1.0)
        else:
            denoised = self._denoise(
                rgb_linear,
                bilateral_diameter,
                bilateral_sigma_color,
                bilateral_sigma_space,
                apply_guided_filter,
                guided_radius,
                guided_eps,
            )

        sr = denoised
        superres_applied = False
        if enable_superres and not exact_palette_mode:
            sr, superres_applied = self._run_super_resolution(
                denoised,
                scale=superres_scale,
                model_path=superres_model_path,
            )

        if exact_palette_mode:
            tone_mapped = np.clip(sr, 0.0, 1.0)
        else:
            tone_mapped = self._apply_clahe_and_sharpen(sr, clahe_clip_limit, clahe_grid_size, sharpen_amount)

        if exact_palette_mode:
            calibrated = np.clip(tone_mapped, 0.0, 1.0)
        else:
            calibrated = self._apply_calibration(tone_mapped, calibration_profile)

        palette = self._reduce_palette(calibrated, palette_size, palette_colors)

        if exact_palette_mode:
            H_exact, W_exact = palette.indexed_image.shape
            dithered = palette.palette_rgb[palette.indexed_image].reshape(H_exact, W_exact, 3)
        else:
            dithered = self._apply_dithering(calibrated, palette, method=dither)
            dithered = self._cleanup_black_speckles(dithered)
            dithered = self._refine_post_slicing(dithered)

        stroke_plan = self._plan_strokes(dithered, palette, slic_segments, slic_compactness, stroke_spacing_px)

        simulation = self._render_simulation(palette, dithered.shape[:2])

        # Run the painterly post process twice to guarantee the full refinement
        # cycle described in the specification.  The second pass operates on the
        # already enhanced output which mimics analysing and repainting the
        # intermediate canvas once more for maximal fidelity.
        if exact_palette_mode:
            post_processed = np.clip(simulation, 0.0, 1.0)
            post_processed_passes: List[np.ndarray] = [post_processed]
        else:
            post_processed_passes = []
            post_pass_input = simulation
            for _ in range(2):
                post_pass_input = self._post_process_render(post_pass_input, reference_rgb=calibrated)
                post_processed_passes.append(post_pass_input)
            post_processed = post_processed_passes[-1] if post_processed_passes else simulation

        metrics = self._evaluate_quality(rgb01, post_processed)

        result = PipelineResult(
            processed_rgb=sr,
            calibrated_rgb=calibrated,
            palette=palette,
            dithered_rgb=dithered,
            stroke_plan=stroke_plan,
            simulation_rgb=simulation,
            post_processed_rgb=post_processed,
            metrics=metrics,
            config={
                "enable_superres": enable_superres and superres_applied,
                "superres_scale": superres_scale,
                "dither": dither,
                "palette_size": palette_size,
                "slic_segments": slic_segments,
                "slic_compactness": slic_compactness,
                "stroke_spacing_px": stroke_spacing_px,
                "calibration_profile": calibration_profile,
                "palette_colors": palette_colors,
                "post_process_passes": len(post_processed_passes) or 0,
                "exact_palette_mode": exact_palette_mode,
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

    @property
    def super_resolution_available(self) -> bool:
        """Return ``True`` if the Real-ESRGAN stack can be used."""

        return self._superres_supported

    def _run_super_resolution(
        self,
        rgb_linear: np.ndarray,
        *,
        scale: int,
        model_path: Optional[str],
    ) -> Tuple[np.ndarray, bool]:
        if not self._superres_supported:
            if not self._superres_warning_emitted:
                LOGGER.warning("Real-ESRGAN not available – skipping super-resolution.")
                self._superres_warning_emitted = True
            return rgb_linear, False

        if self._realesrgan is None:
            model = _sr_default_model(scale)
            if model is None:
                if not self._superres_warning_emitted:
                    LOGGER.warning("RRDBNet arch unavailable – skipping super-resolution.")
                    self._superres_warning_emitted = True
                return rgb_linear, False

            model_path = model_path or os.environ.get("PAINTER_REAL_ESRGAN_MODEL")
            if not model_path or not os.path.exists(model_path):
                if not self._superres_weights_warning_emitted:
                    LOGGER.warning("Real-ESRGAN model weights missing – skipping super-resolution.")
                    self._superres_weights_warning_emitted = True
                return rgb_linear, False

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
            return rgb_linear, False

        rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return _linearise_srgb(np.clip(rgb, 0.0, 1.0)), True

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

        rgb_out = _lab_to_srgb_safe(lab)
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
        elif palette_size is None or palette_size <= 0 or palette_size >= H * W:
            palette_rgb = rgb.reshape(-1, 3).copy()
            indexed_image = np.arange(H * W, dtype=np.int32).reshape(H, W)
            delta_e_map = np.zeros((H, W), dtype=np.float32)
            return PaletteResult(
                palette_rgb=palette_rgb,
                indexed_image=indexed_image,
                delta_e_map=delta_e_map,
            )
        else:
            palette = self._palette_via_libimagequant(img_uint8, palette_size)
            if palette is None:
                palette = self._palette_via_kmeans(rgb_linear, palette_size)

        palette = palette.reshape(-1, 3)

        value_channel = np.max(rgb, axis=2)
        black_pixel_fraction = float(np.mean(value_channel < 0.08))
        allow_true_black = black_pixel_fraction >= 0.02

        palette_lab = skcolor.rgb2lab(palette.reshape(-1, 1, 1, 3)).reshape(-1, 3)
        if not allow_true_black:
            min_l_star = 8.0
            dark_mask = palette_lab[:, 0] < min_l_star
            if np.any(dark_mask):
                palette_lab = palette_lab.copy()
                palette_lab[dark_mask, 0] = min_l_star
                palette = np.clip(
                    _lab_to_srgb_safe(palette_lab.reshape(-1, 1, 1, 3)).reshape(-1, 3),
                    0.0,
                    1.0,
                )
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
        error_scale = 0.95
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
                        out[ny, nx] = np.clip(
                            out[ny, nx] + quant_error * weight * error_scale,
                            0.0,
                            1.0,
                        )

        return np.clip(out, 0.0, 1.0)

    def _cleanup_black_speckles(
        self,
        rgb: np.ndarray,
        *,
        value_threshold: float = 0.08,
        min_component_size: int = 36,
        inpaint_radius: int = 3,
    ) -> np.ndarray:
        srgb = np.clip(rgb, 0.0, 1.0)
        value = np.max(srgb, axis=2)
        candidate_mask = (value < value_threshold).astype(np.uint8)

        if np.count_nonzero(candidate_mask) == 0:
            return srgb

        kernel = np.ones((3, 3), np.uint8)
        num_labels, labels = cv2.connectedComponents(candidate_mask)
        removal_mask = np.zeros_like(candidate_mask, dtype=np.uint8)

        for label in range(1, num_labels):
            component = labels == label
            component_size = int(np.count_nonzero(component))
            if component_size >= min_component_size:
                continue

            dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
            border = np.logical_and(dilated, ~component)
            if np.count_nonzero(border) == 0:
                continue

            border_value = value[border]
            if float(np.mean(border_value)) < max(value_threshold * 2.0, 0.18):
                continue

            removal_mask[dilated] = 255

        if np.count_nonzero(removal_mask) == 0:
            cleaned_rgb = srgb
        else:
            inpaint_input = (srgb * 255).astype(np.uint8)
            inpaint_bgr = cv2.cvtColor(inpaint_input, cv2.COLOR_RGB2BGR)
            cleaned_bgr = cv2.inpaint(inpaint_bgr, removal_mask, inpaint_radius, cv2.INPAINT_TELEA)
            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        residual_mask = cleaned_rgb.max(axis=2) < value_threshold
        if np.count_nonzero(residual_mask) == 0:
            return np.clip(cleaned_rgb, 0.0, 1.0)

        median_rgb = cv2.medianBlur((cleaned_rgb * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
        cleaned_rgb[residual_mask] = median_rgb[residual_mask]
        return np.clip(cleaned_rgb, 0.0, 1.0)

    def _refine_post_slicing(
        self,
        dithered_rgb: np.ndarray,
        *,
        closing_iterations: int = 2,
        highlight_percentile: float = 0.87,
        highlight_strength: float = 0.18,
    ) -> np.ndarray:
        """Final pass after slicing to tidy pepper noise and softly lift highlights."""

        srgb = np.clip(dithered_rgb, 0.0, 1.0).astype(np.float32)
        if srgb.size == 0:
            return srgb

        img8 = (srgb * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
        value = hsv[..., 2]

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(value, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(closing_iterations)))
        blended = cv2.addWeighted(value, 0.45, closed, 0.55, 0)
        blended = cv2.medianBlur(blended, 3)
        hsv[..., 2] = blended
        refined_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        residual_dark = refined_rgb.max(axis=2) < 0.12
        if np.count_nonzero(residual_dark):
            residual_mask = cv2.dilate(residual_dark.astype(np.uint8) * 255, kernel, iterations=1)
            inpaint_bgr = cv2.cvtColor((refined_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(inpaint_bgr, residual_mask, 3, cv2.INPAINT_TELEA)
            refined_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        lab = cv2.cvtColor((np.clip(refined_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
        L_channel = lab[..., 0].astype(np.float32) / 255.0
        percentile = float(np.clip(highlight_percentile, 0.0, 1.0) * 100.0)
        threshold = np.percentile(L_channel, percentile)
        highlight_mask = L_channel >= threshold

        if np.any(highlight_mask) and np.count_nonzero(highlight_mask) < highlight_mask.size:
            expanded = cv2.dilate((highlight_mask.astype(np.uint8) * 255), kernel, iterations=1).astype(bool)
            strength = float(np.clip(highlight_strength, 0.0, 1.0))
            L_channel[expanded] = np.clip(
                L_channel[expanded] + strength * (1.0 - L_channel[expanded]),
                0.0,
                1.0,
            )
            lab[..., 0] = (L_channel * 255.0).astype(np.uint8)

        highlighted_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return np.clip(highlighted_rgb, 0.0, 1.0)

    def _post_process_render(
        self,
        simulation_rgb: np.ndarray,
        *,
        reference_rgb: Optional[np.ndarray] = None,
        highlight_boost: float = 0.18,
        shadow_boost: float = 0.14,
        saturation_boost: float = 0.1,
    ) -> np.ndarray:
        """High quality enhancement pass that emulates a painterly post process."""

        srgb = np.clip(simulation_rgb, 0.0, 1.0).astype(np.float32)
        if srgb.size == 0:
            return srgb

        reference = srgb if reference_rgb is None else np.clip(reference_rgb, 0.0, 1.0).astype(np.float32)

        black_mask = srgb.max(axis=2) < 0.04
        if np.any(black_mask):
            blurred_reference = cv2.GaussianBlur((reference * 255.0).astype(np.uint8), (0, 0), 1.8)
            blurred_reference = blurred_reference.astype(np.float32) / 255.0
            srgb[black_mask] = blurred_reference[black_mask]

        ref_gray = cv2.cvtColor((reference * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(ref_gray, 40, 140)
        structure = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 1.2)
        structure = np.clip(structure, 0.0, 1.0)

        lab = cv2.cvtColor((srgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        L_channel = lab[..., 0] / 255.0

        highlight_threshold = np.percentile(L_channel, 85.0)
        highlight_map = np.clip((L_channel - highlight_threshold) / max(1e-4, 1.0 - highlight_threshold), 0.0, 1.0)
        highlight_map = cv2.GaussianBlur(highlight_map.astype(np.float32), (0, 0), 1.1)
        L_channel = np.clip(L_channel + float(np.clip(highlight_boost, 0.0, 1.0)) * highlight_map, 0.0, 1.0)

        shadow_threshold = np.percentile(L_channel, 35.0)
        shadow_map = np.clip((shadow_threshold - L_channel) / max(1e-4, shadow_threshold), 0.0, 1.0)
        shadow_map = cv2.GaussianBlur(shadow_map.astype(np.float32), (0, 0), 1.6)
        L_channel = np.clip(
            L_channel - float(np.clip(shadow_boost, 0.0, 1.0)) * shadow_map * (1.0 - structure * 0.75),
            0.0,
            1.0,
        )

        lab[..., 0] = np.clip(L_channel * 255.0, 0.0, 255.0)
        tonemapped = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        hsv = cv2.cvtColor((tonemapped * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[..., 1] / 255.0
        ref_hsv = cv2.cvtColor((reference * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        ref_saturation = ref_hsv[..., 1] / 255.0
        target_saturation = np.percentile(ref_saturation, 75.0)
        sat_gain = float(np.clip(saturation_boost, 0.0, 1.0)) * np.clip(target_saturation - saturation, 0.0, 1.0)
        saturation = np.clip(saturation + sat_gain + structure * 0.08, 0.0, 1.0)
        hsv[..., 1] = saturation * 255.0

        value = hsv[..., 2] / 255.0
        value = np.clip(value + highlight_map * 0.4 * float(np.clip(highlight_boost, 0.0, 1.0)), 0.0, 1.0)
        hsv[..., 2] = value * 255.0

        enhanced = cv2.cvtColor(np.clip(hsv, 0.0, 255.0).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        overlay = cv2.GaussianBlur(structure, (0, 0), 1.0)[..., None]
        enhanced = np.clip(enhanced + overlay * 0.05, 0.0, 1.0)

        refined = cv2.bilateralFilter((enhanced * 255.0).astype(np.uint8), d=5, sigmaColor=35, sigmaSpace=9)
        refined = refined.astype(np.float32) / 255.0
        return np.clip(refined, 0.0, 1.0)

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

        lpips_val = float("nan")
        if lpips is not None and torch is not None:
            if self._lpips_model is None:
                self._lpips_model = _initialise_lpips_model()
            if self._lpips_model is not None:
                ref_tensor = torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
                sim_tensor = torch.from_numpy(sim.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
                with torch.no_grad():  # pragma: no cover - requires torch
                    lpips_val = float(self._lpips_model(ref_tensor, sim_tensor).item())
        metrics["lpips"] = lpips_val

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

