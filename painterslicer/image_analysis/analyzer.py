import hashlib
import os
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import slic
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage as ndi

try:  # SimpleITK ist optional, verbessert aber die Rauschunterdrückung massiv
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - Fallback erlaubt weiterhin Basisfunktionalität
    sitk = None


def _fit_kmeans_compat(
    data: np.ndarray,
    *,
    n_clusters: int,
    random_state: int = 0,
) -> KMeans:
    """Fit ``KMeans`` with ``n_init`` compatible across scikit-learn versions."""

    try:
        model = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=random_state,
        )
        return model.fit(data)
    except (TypeError, ValueError):
        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
        )
        return model.fit(data)


class ImageAnalyzer:
    """Bildanalyse / Vorverarbeitung."""

    def __init__(self):
        self.last_color_analysis: Optional[Dict[str, Any]] = None
        self.last_layer_analysis: Optional[Dict[str, Any]] = None
        self.last_enhanced_rgb01: Optional[np.ndarray] = None
        self.last_enhanced_bgr: Optional[np.ndarray] = None
        self._last_enhanced_signature: Optional[Tuple[Any, ...]] = None

    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        return img

    def edge_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        return edges

    def analyze_for_preview(self, image_path: str) -> np.ndarray:
        img = self.enhance_image_quality(image_path)
        if img is None:
            img = self.load_image(image_path)
        mask = self.edge_mask(img)
        return mask

    # ----------------------------
    #  NEU: Layer-Extraktion
    # ----------------------------
    def _make_source_signature(
        self, image_source: Union[str, np.ndarray]
    ) -> Tuple[Any, ...]:
        if isinstance(image_source, str):
            path = os.path.abspath(image_source)
            try:
                stat = os.stat(path)
                mtime = int(stat.st_mtime)
                size = int(stat.st_size)
            except OSError:
                mtime = None
                size = None
            return ("path", path, mtime, size)

        arr = np.asarray(image_source)
        shape = tuple(arr.shape)
        dtype = str(arr.dtype)
        if arr.size == 0:
            return ("array", shape, dtype, 0)

        flat = arr.reshape(-1)
        sample = flat[: min(flat.size, 4096)].tobytes()
        digest = hashlib.sha1(sample).hexdigest()
        return ("array", shape, dtype, digest, int(arr.size))

    def enhance_image_quality(
        self,
        image_source: Union[str, np.ndarray],
        *,
        color_boost: float = 1.05,
        contrast_boost: float = 1.12,
        sharpness_boost: float = 1.15,
        anisotropic_iterations: int = 12,
        anisotropic_conductance: float = 1.8,
        gaussian_sigma: float = 0.8,
        detail_amount: float = 0.55,
        median_size: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Führt eine hochwertige Vorverarbeitung durch, die mehrere moderne Bibliotheken
        kombiniert (Pillow, SimpleITK, SciPy). Das Ergebnis ist ein kontrastreiches
        und entrauschtes BGR-Bild, das insbesondere für das Layering und die
        Segmentierung optimiert ist.

        Die einzelnen Schritte basieren auf bewährten Workflows aus der
        Bildverarbeitung: sanfte Farbanpassung (Pillow), anisotrope Diffusion zur
        Rauschunterdrückung bei gleichzeitiger Kantenerhaltung (SimpleITK) und ein
        unsharp-masking-ähnlicher Boost der Details (SciPy ndimage).
        """

        source_signature = self._make_source_signature(image_source)

        if (
            self.last_enhanced_bgr is not None
            and self._last_enhanced_signature == source_signature
        ):
            return self.last_enhanced_bgr.copy()

        base_bgr = self._ensure_bgr_uint8(image_source)
        if base_bgr is None:
            return None

        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
        rgb01 = (base_rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)

        pil_img = Image.fromarray((rgb01 * 255).astype(np.uint8))
        pil_img = ImageEnhance.Color(pil_img).enhance(color_boost)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_boost)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness_boost)
        pil_img = pil_img.filter(ImageFilter.DETAIL)
        pil_arr = np.asarray(pil_img).astype(np.float32) / 255.0

        processed = pil_arr
        if sitk is not None:
            diffusion_kwargs = dict(
                timeStep=0.05,
                conductanceParameter=float(np.clip(anisotropic_conductance, 0.1, 5.0)),
                numberOfIterations=int(np.clip(anisotropic_iterations, 1, 30)),
            )

            try:
                sitk_img = sitk.GetImageFromArray(
                    processed.astype(np.float32), isVector=True
                )
                sitk_img = sitk.CurvatureAnisotropicDiffusion(
                    sitk_img,
                    **diffusion_kwargs,
                )
                processed = sitk.GetArrayFromImage(sitk_img)
            except RuntimeError:
                # Einige SimpleITK Builds (v.a. auf Windows) unterstützen die
                # Vektorvariante nicht. In diesem Fall wenden wir die Diffusion
                # kanalweise an und setzen das Ergebnis wieder zusammen.
                channel_results: List[np.ndarray] = []
                for channel in range(processed.shape[-1]):
                    channel_img = sitk.GetImageFromArray(
                        processed[..., channel].astype(np.float32)
                    )
                    channel_img = sitk.CurvatureAnisotropicDiffusion(
                        channel_img,
                        **diffusion_kwargs,
                    )
                    channel_arr = sitk.GetArrayFromImage(channel_img)
                    channel_results.append(np.asarray(channel_arr, dtype=np.float32))

                processed = np.stack(channel_results, axis=-1)

            processed = np.asarray(processed, dtype=np.float32)
            processed = np.clip(processed, 0.0, 1.0)

        gaussian_sigma = float(np.clip(gaussian_sigma, 0.2, 3.0))
        median_size = int(max(3, median_size if median_size % 2 == 1 else median_size + 1))
        detail_amount = float(np.clip(detail_amount, 0.0, 1.0))

        blurred = np.stack(
            [
                ndi.gaussian_filter(processed[..., c], sigma=gaussian_sigma, mode="reflect")
                for c in range(processed.shape[-1])
            ],
            axis=-1,
        )

        baseline = np.stack(
            [
                ndi.median_filter(blurred[..., c], size=median_size, mode="reflect")
                for c in range(blurred.shape[-1])
            ],
            axis=-1,
        )

        boosted = np.clip(blurred + detail_amount * (blurred - baseline), 0.0, 1.0)

        gray_edges = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
        edge_strength = cv2.Canny(gray_edges, 60, 160)
        edge_weight = cv2.GaussianBlur(edge_strength.astype(np.float32) / 255.0, (5, 5), 0)
        edge_weight = edge_weight[..., None]

        mixed = np.clip(
            boosted * (1.0 - 0.35 * edge_weight) + rgb01 * (0.35 * edge_weight),
            0.0,
            1.0,
        )

        final_rgb01 = mixed.astype(np.float32)
        final_rgb_u8 = np.clip(final_rgb01 * 255.0, 0, 255).astype(np.uint8)
        final_bgr = cv2.cvtColor(final_rgb_u8, cv2.COLOR_RGB2BGR)

        self.last_enhanced_rgb01 = final_rgb01.copy()
        self.last_enhanced_bgr = final_bgr.copy()
        self._last_enhanced_signature = source_signature

        return self.last_enhanced_bgr.copy()

    def _ensure_bgr_uint8(
        self, image_source: Union[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Internal helper that converts an arbitrary image input into a BGR uint8 array.

        ``image_source`` can either be a file path or an RGB-like numpy array. Arrays are
        assumed to be in RGB channel order if they are not already BGR; they will be
        converted to BGR for OpenCV processing.
        """

        if isinstance(image_source, str):
            return self.load_image(image_source)

        img_arr = np.asarray(image_source)
        if img_arr.ndim != 3 or img_arr.shape[2] != 3:
            raise ValueError("expect image as (H,W,3) array")

        if img_arr.dtype == np.uint8:
            img_uint8 = img_arr.copy()
        else:
            arr = img_arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            img_uint8 = np.clip(arr, 0, 255).astype(np.uint8)

        try:
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        except cv2.error:
            # Falls das Eingabebild schon im BGR-Format war, führt eine erneute
            # Umwandlung manchmal zu Fehlern. Dann versuchen wir es ohne Konvertierung.
            img_bgr = img_uint8

        return img_bgr

    def _normalize_map(self, arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return None

        arr = arr.astype(np.float32)
        min_val = float(arr.min(initial=0.0))
        max_val = float(arr.max(initial=0.0))
        if max_val - min_val < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_val) / (max_val - min_val)

    def _analyze_layers_with_opencv(
        self, img_bgr: np.ndarray
    ) -> Dict[str, Any]:
        """
        Erstellt mehrstufige Schichtenmasken mithilfe umfangreicher OpenCV-Verarbeitung.

        Das Verfahren kombiniert:
          - edgePreservingFilter / bilateralFilter für Farbrauschen-Reduktion
          - multi-scale Gradienten (Scharr, Laplacian)
          - Gabor-Filter zur Texturerkennung
          - Canny-Kanten und Distanztransform zur Tiefen-Heuristik
          - Morphologische Operationen zur Masken-Verfeinerung

        Rückgabe:
            {
                "masks": {...},
                "score_maps": {...},
                "edge_maps": {...},
                "preprocessed": <BGR-Image>,
            }
        """

        if img_bgr is None:
            return {
                "masks": {
                    "background": None,
                    "mid": None,
                    "detail": None,
                },
                "score_maps": {},
                "edge_maps": {},
                "preprocessed": None,
            }

        H, W = img_bgr.shape[:2]

        try:
            pre = cv2.edgePreservingFilter(img_bgr, flags=1, sigma_s=40, sigma_r=0.3)
        except cv2.error:
            pre = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=90, sigmaSpace=45)

        blur_small = cv2.GaussianBlur(pre, (5, 5), 0)
        blur_large = cv2.GaussianBlur(pre, (15, 15), 3.0)

        gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.cvtColor(blur_small, cv2.COLOR_BGR2GRAY)
        gray_large = cv2.cvtColor(blur_large, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(scharr_x, scharr_y)

        laplace_small = cv2.Laplacian(gray_small, cv2.CV_32F, ksize=3)
        laplace_large = cv2.Laplacian(gray_large, cv2.CV_32F, ksize=5)
        laplace_mix = np.abs(laplace_small) + 0.6 * np.abs(laplace_large)

        gabor_response = np.zeros_like(gray_f, dtype=np.float32)
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            kernel = cv2.getGaborKernel((11, 11), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
            gabor_response = np.maximum(gabor_response, np.abs(filtered))

        edges_detail = cv2.Canny(gray, 70, 190)
        edges_mid = cv2.Canny(gray_small, 40, 140)
        edges_soft = cv2.Canny(gray_large, 25, 90)
        edges_combined = cv2.max(cv2.max(edges_detail, edges_mid), edges_soft)

        grad_norm = self._normalize_map(grad_mag)
        lap_norm = self._normalize_map(laplace_mix)
        gabor_norm = self._normalize_map(gabor_response)
        edges_norm = self._normalize_map(edges_combined.astype(np.float32))

        detail_score = np.clip(
            0.55 * grad_norm + 0.35 * edges_norm + 0.20 * lap_norm,
            0.0,
            1.0,
        )

        texture_score = np.clip(
            0.5 * lap_norm + 0.3 * gabor_norm + 0.2 * grad_norm,
            0.0,
            1.0,
        )

        edges_dilated = cv2.dilate(edges_mid, np.ones((5, 5), np.uint8))
        inv_edges = cv2.bitwise_not(edges_dilated)
        dist_map = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
        dist_norm = self._normalize_map(dist_map)

        hsv = cv2.cvtColor(pre, cv2.COLOR_BGR2HSV)
        saturation_norm = self._normalize_map(hsv[:, :, 1])

        background_score = np.clip(
            0.6 * dist_norm + 0.2 * (1.0 - texture_score) + 0.2 * (1.0 - saturation_norm),
            0.0,
            1.0,
        )

        mid_score = np.clip(
            0.55 * texture_score + 0.25 * (1.0 - dist_norm) + 0.20 * saturation_norm,
            0.0,
            1.0,
        )

        detail_score = np.clip(
            0.65 * detail_score + 0.20 * (1.0 - dist_norm) + 0.15 * saturation_norm,
            0.0,
            1.0,
        )

        score_stack = np.stack([
            background_score,
            mid_score,
            detail_score,
        ], axis=-1)

        assignment = np.argmax(score_stack, axis=-1).astype(np.uint8)

        background_mask = (assignment == 0).astype(np.uint8) * 255
        mid_mask = (assignment == 1).astype(np.uint8) * 255
        detail_mask = (assignment == 2).astype(np.uint8) * 255

        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        detail_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, bg_kernel)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, mid_kernel)

        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_CLOSE, mid_kernel)
        mid_mask = cv2.morphologyEx(mid_mask, cv2.MORPH_OPEN, detail_kernel)

        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_DILATE, detail_kernel)
        detail_mask = cv2.bitwise_and(detail_mask, edges_combined)

        mid_mask = cv2.bitwise_and(
            mid_mask,
            cv2.bitwise_not(detail_mask),
        )
        background_mask = cv2.bitwise_and(
            background_mask,
            cv2.bitwise_not(cv2.bitwise_or(mid_mask, detail_mask)),
        )

        masks = {
            "background": background_mask,
            "mid": mid_mask,
            "detail": detail_mask,
        }

        return {
            "masks": masks,
            "score_maps": {
                "background": background_score.astype(np.float32),
                "mid": mid_score.astype(np.float32),
                "detail": detail_score.astype(np.float32),
                "texture": texture_score.astype(np.float32),
                "distance": dist_norm.astype(np.float32),
            },
            "edge_maps": {
                "detail": edges_detail,
                "mid": edges_mid,
                "combined": edges_combined,
            },
            "preprocessed": pre,
            "assignment": assignment,
        }

    def make_layer_masks(
        self, image_source: Union[str, np.ndarray]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Verwendet eine erweiterte OpenCV-Pipeline, um drei Schichtenmasken
        (Hintergrund, Mittelgrund, Detail) für das Malprogramm zu erzeugen.
        """

        enhanced_bgr = self.enhance_image_quality(image_source)
        if enhanced_bgr is None:
            enhanced_bgr = self._ensure_bgr_uint8(image_source)

        analysis = self._analyze_layers_with_opencv(enhanced_bgr)
        self.last_layer_analysis = analysis

        masks = analysis.get("masks", {})

        return {
            "background_mask": masks.get("background"),
            "mid_mask": masks.get("mid"),
            "detail_mask": masks.get("detail"),
        }

    def plan_painting_layers(
        self,
        image_source: Union[str, np.ndarray],
        *,
        k_colors: Optional[int] = None,
        k_min: int = 8,
        k_max: int = 16,
        style_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Erzeugt einen heuristischen Malplan von groben zu feinen Schichten.

        Rückgabeformat::

            {
                "image_size": (H, W),
                "layer_masks": {"background_mask": ..., ...},
                "layers": [
                    {
                        "label": int,
                        "color_rgb": (r, g, b),
                        "coverage": float,
                        "stage": "background" | "mid" | "detail",
                        "tool": str,
                        "technique": str,
                        "detail_ratio": float,
                        "mid_ratio": float,
                        "background_ratio": float,
                        "path_count": int,
                        "path_length": int,
                        "order": int,
                        "pixel_paths": [...]
                    },
                    ...
                ],
            }

        Damit lässt sich das Bild von hinten (große Farbflächen) nach vorne (Details)
        planen und jeweils ein geeignetes Werkzeug auswählen.
        """

        enhanced_bgr = self.enhance_image_quality(image_source)
        if enhanced_bgr is not None and self.last_enhanced_rgb01 is not None:
            img_rgb01 = self.last_enhanced_rgb01.copy()
        else:
            img_rgb01 = self._ensure_rgb01(image_source)
            img_rgb01 = img_rgb01.copy()
        img_lab01 = rgb2lab(img_rgb01)
        H, W = img_rgb01.shape[:2]

        layer_masks = self.make_layer_masks(image_source)

        score_maps = {}
        if self.last_layer_analysis is not None:
            score_maps = self.last_layer_analysis.get("score_maps", {})
        detail_score_map = score_maps.get("detail")
        mid_score_map = score_maps.get("mid")
        background_score_map = score_maps.get("background")
        texture_score_map = score_maps.get("texture")

        style_profile = style_profile or {}

        extract_kwargs: Dict[str, Any] = {
            "k_colors": style_profile.get("k_colors", k_colors),
            "k_min": style_profile.get("k_min", k_min),
            "k_max": style_profile.get("k_max", k_max),
            "use_dither": style_profile.get("use_dither", True),
            "min_path_length": style_profile.get("min_path_length", 2),
            "min_area_ratio": style_profile.get("min_area_ratio", 0.0005),
            "stroke_spacing_scale": style_profile.get("stroke_spacing_scale", 1.0),
            "preserve_edge_strokes": style_profile.get("preserve_edge_strokes", False),
            "detail_edge_boost": style_profile.get("detail_edge_boost", 1.0),
            "edge_sensitivity": style_profile.get("edge_sensitivity", 1.0),
            "microtransition_boost": style_profile.get("microtransition_boost", 1.0),
            "chroma_boost": style_profile.get("chroma_boost", 1.0),
            "highlight_boost": style_profile.get("highlight_boost", 0.0),
        }

        color_layers = self.extract_color_layers(
            image_source,
            **extract_kwargs,
        )

        labels = None
        if self.last_color_analysis is not None:
            labels = self.last_color_analysis.get("labels")

        if labels is None:
            raise RuntimeError(
                "labels not available after color extraction; did extract_color_layers fail?"
            )

        if labels.shape[0] != H or labels.shape[1] != W:
            labels = cv2.resize(
                labels.astype(np.int32),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )

        background_mask = layer_masks.get("background_mask")
        mid_mask = layer_masks.get("mid_mask")
        detail_mask = layer_masks.get("detail_mask")

        if background_mask is not None and background_mask.shape != (H, W):
            background_mask = cv2.resize(background_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if mid_mask is not None and mid_mask.shape != (H, W):
            mid_mask = cv2.resize(mid_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if detail_mask is not None and detail_mask.shape != (H, W):
            detail_mask = cv2.resize(detail_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        total_pixels = float(H * W)

        def _mask_overlap_ratio(layer_mask: np.ndarray, ref_mask: Optional[np.ndarray]) -> float:
            if ref_mask is None or ref_mask.max() == 0:
                return 0.0
            ref_bool = ref_mask > 0
            overlap = np.logical_and(layer_mask, ref_bool)
            layer_area = max(int(np.count_nonzero(layer_mask)), 1)
            return float(np.count_nonzero(overlap)) / float(layer_area)

        def _mean_score(score_map: Optional[np.ndarray], mask_bool: np.ndarray) -> float:
            if score_map is None:
                return 0.0
            values = score_map[mask_bool]
            if values.size == 0:
                return 0.0
            return float(np.mean(values))

        stage_priority = {"background": 0, "mid": 1, "detail": 2}

        planned_layers: List[Dict[str, Any]] = []

        highlight_bias = float(np.clip(style_profile.get("highlight_bias", 1.0), 0.2, 3.0))
        shadow_bias = float(np.clip(style_profile.get("shadow_bias", 1.0), 0.2, 3.0))
        color_variance_bias = float(
            np.clip(style_profile.get("color_variance_bias", 1.0), 0.2, 3.0)
        )

        for layer in color_layers:
            label = int(layer["label"])
            layer_mask = labels == label
            layer_area = int(np.count_nonzero(layer_mask))
            if layer_area == 0:
                continue

            coverage = layer_area / total_pixels
            detail_ratio = _mask_overlap_ratio(layer_mask, detail_mask)
            mid_ratio = _mask_overlap_ratio(layer_mask, mid_mask)
            background_ratio = _mask_overlap_ratio(layer_mask, background_mask)

            detail_strength = _mean_score(detail_score_map, layer_mask)
            mid_strength = _mean_score(mid_score_map, layer_mask)
            background_strength = _mean_score(background_score_map, layer_mask)
            texture_strength = _mean_score(texture_score_map, layer_mask)

            highlight_strength = 0.0
            shadow_strength = 0.0
            contrast_strength = 0.0
            color_variance_strength = 0.0

            if np.any(layer_mask):
                layer_lab = img_lab01[layer_mask]
                if layer_lab.size:
                    l_vals = layer_lab[:, 0] / 100.0
                    if l_vals.size > 0:
                        if l_vals.size >= 5:
                            high_pct = float(np.percentile(l_vals, 90))
                            low_pct = float(np.percentile(l_vals, 10))
                        else:
                            high_pct = float(np.max(l_vals))
                            low_pct = float(np.min(l_vals))

                        high_frac = float(np.mean(l_vals > 0.75))
                        low_frac = float(np.mean(l_vals < 0.35))

                        highlight_strength = float(
                            np.clip(
                                0.6 * high_frac
                                + 0.4 * max(0.0, high_pct - 0.65) / 0.35,
                                0.0,
                                1.0,
                            )
                        )
                        shadow_strength = float(
                            np.clip(
                                0.6 * low_frac
                                + 0.4 * max(0.0, 0.45 - low_pct) / 0.45,
                                0.0,
                                1.0,
                            )
                        )
                        contrast_strength = float(
                            np.clip((high_pct - low_pct) / 0.5, 0.0, 1.0)
                        )

                    ab = layer_lab[:, 1:]
                    if ab.size:
                        chroma = np.sqrt(np.square(ab[:, 0]) + np.square(ab[:, 1]))
                        chroma_std = float(np.std(chroma))
                        ab_std = float(np.mean(np.std(ab, axis=0)))
                        color_variance_strength = float(
                            np.clip(
                                0.5 * (chroma_std / 25.0)
                                + 0.5 * (ab_std / 20.0),
                                0.0,
                                1.0,
                            )
                        )

            highlight_strength = float(
                np.clip(highlight_strength * highlight_bias, 0.0, 1.0)
            )
            shadow_strength = float(np.clip(shadow_strength * shadow_bias, 0.0, 1.0))
            color_variance_strength = float(
                np.clip(color_variance_strength * color_variance_bias, 0.0, 1.0)
            )

            ratios = {
                "background": 0.5 * background_ratio + 0.5 * background_strength,
                "mid": 0.5 * mid_ratio + 0.5 * mid_strength,
                "detail": 0.5 * detail_ratio + 0.5 * detail_strength,
            }
            stage_scores = {
                "background": ratios["background"]
                + 0.2 * max(0.0, 0.4 - highlight_strength)
                + 0.15 * max(0.0, 0.5 - color_variance_strength),
                "mid": ratios["mid"]
                + 0.15 * color_variance_strength
                + 0.1 * texture_strength
                + 0.1 * shadow_strength,
                "detail": ratios["detail"]
                + 0.25 * highlight_strength
                + 0.2 * contrast_strength
                + 0.15 * color_variance_strength,
            }
            stage_scores["background"] *= float(style_profile.get("background_stage_gain", 1.0))
            stage_scores["mid"] *= float(style_profile.get("mid_stage_gain", 1.0))
            stage_scores["detail"] *= float(style_profile.get("detail_stage_gain", 1.0))
            stage = max(stage_scores, key=stage_scores.get)

            path_count = len(layer["pixel_paths"])
            path_length = int(
                sum(len(path) for path in layer["pixel_paths"])
            )
            density = float(path_length) / float(layer_area)
            density = float(np.clip(density, 0.0, 1.0))

            tool = "round_brush"
            technique = "layered_strokes"

            if highlight_strength > 0.65 and detail_ratio > 0.3:
                tool = "fine_brush"
                technique = "luminous_glazing"
            elif shadow_strength > 0.6 and coverage < 0.2:
                tool = "flat_brush"
                technique = "shadow_glaze"
            elif coverage > 0.35 and detail_ratio < 0.25 and shadow_strength < 0.5:
                tool = "wide_brush"
                technique = "broad_fill"
            elif stage == "mid" and 0.08 < coverage < 0.25 and detail_ratio < 0.2:
                tool = "sponge"
                technique = "dabbing"
            elif detail_ratio > 0.55 or density > 0.6 or coverage < 0.05 or detail_strength > 0.55:
                tool = "fine_brush"
                technique = "precision_strokes"
            elif (
                stage == "mid"
                and detail_ratio < 0.45
                and color_variance_strength > 0.5
            ):
                tool = "round_brush"
                technique = "vibrant_impasto"
            elif stage == "mid" and detail_ratio < 0.45:
                tool = "flat_brush"
                technique = "feathering"

            if stage == "background" and background_strength > 0.6 and coverage > 0.2:
                technique = "gradient_blend"
            if stage == "detail" and texture_strength > 0.5 and coverage > 0.1:
                technique = "cross_hatching"

            planned_layers.append({
                "label": label,
                "color_rgb": layer["color_rgb"],
                "coverage": float(coverage),
                "stage": stage,
                "tool": tool,
                "technique": technique,
                "detail_ratio": float(detail_ratio),
                "mid_ratio": float(mid_ratio),
                "background_ratio": float(background_ratio),
                "detail_strength": float(detail_strength),
                "mid_strength": float(mid_strength),
                "background_strength": float(background_strength),
                "texture_strength": float(texture_strength),
                "highlight_strength": float(highlight_strength),
                "shadow_strength": float(shadow_strength),
                "contrast_strength": float(contrast_strength),
                "color_variance_strength": float(color_variance_strength),
                "path_count": int(path_count),
                "path_length": int(path_length),
                "pixel_paths": layer["pixel_paths"],
                "stage_priority": stage_priority.get(stage, 1),
            })

        planned_layers.sort(
            key=lambda entry: (
                entry["stage_priority"],
                -entry["background_ratio"],
                entry["coverage"],
            )
        )

        for order, layer in enumerate(planned_layers):
            layer["order"] = order
            layer.pop("stage_priority", None)

        return {
            "image_size": (H, W),
            "layer_masks": {
                "background_mask": background_mask,
                "mid_mask": mid_mask,
                "detail_mask": detail_mask,
            },
            "layers": planned_layers,
        }

    def extract_stroke_paths_from_detail(self, image_path: str):
        """
        Nimmt das Eingabebild, berechnet die detail_mask
        und wandelt die hellen Linienbereiche in Fahrpfade um.

        Rückgabe:
            Eine Liste von Pfaden.
            Jeder Pfad ist eine Liste von (x, y)-Tupeln in Bildkoordinaten.

            Beispiel:
            [
              [(10, 20), (11, 21), (12, 22), ...],
              [(100, 50), (102, 53), (105, 60), ...],
              ...
            ]
        """

        masks = self.make_layer_masks(image_path)
        detail_mask = masks.get("detail_mask")

        if detail_mask is None:
            return []

        if detail_mask.max() == 0:
            return []

        contours, _ = cv2.findContours(
            detail_mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        paths = []

        for cnt in contours:
            cnt = cnt.squeeze(axis=1)
            if len(cnt.shape) != 2 or cnt.shape[0] < 2:
                continue

            path = []
            for (x, y) in cnt:
                path.append((int(x), int(y)))

            paths.append(path)

        return paths

    def extract_paths_from_mask(self, mask: np.ndarray):
        """Nimmt eine Binärmaske (0/255 uint8) und gibt Stroke-Pfade zurück."""
        contours, _ = cv2.findContours(
            mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        paths = []
        for cnt in contours:
            cnt = cnt.squeeze(axis=1)
            if len(cnt.shape) != 2 or cnt.shape[0] < 2:
                continue
            path = []
            for (x, y) in cnt:
                path.append((int(x), int(y)))
            paths.append(path)
        return paths

    def mask_to_offset_contours(
        self,
        mask: np.ndarray,
        *,
        spacing_px: int = 3,
        min_len: int = 5,
        max_loops: int = 512,
    ) -> List[List[Tuple[int, int]]]:
        """
        Erodiert eine Maske schrittweise und sammelt pro Iteration die äußeren Konturen.

        Ergebnis ist eine Liste von Pfaden, die die Fläche in nahezu gleichmäßigen
        Abständen überdecken – damit entstehen im Slicer echte Füllstrukturen statt
        nur Randlinien.
        """

        if mask is None or mask.size == 0:
            return []

        spacing_px = int(max(1, spacing_px))
        if spacing_px % 2 == 0:
            spacing_px += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (spacing_px, spacing_px),
        )
        if kernel is None:
            kernel = np.ones((3, 3), np.uint8)

        current = mask.copy()
        if current.dtype != np.uint8:
            current = current.astype(np.uint8)

        paths: List[List[Tuple[int, int]]] = []
        loop_idx = 0

        while loop_idx < max_loops and np.any(current):
            contours, _ = cv2.findContours(
                current,
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )

            for cnt in contours:
                cnt = cnt.squeeze(axis=1)
                if len(cnt.shape) != 2 or cnt.shape[0] < min_len:
                    continue

                path = [(int(x), int(y)) for (x, y) in cnt]
                if loop_idx % 2 == 1:
                    path.reverse()
                paths.append(path)

            eroded = cv2.erode(current, kernel, iterations=1)
            if np.array_equal(eroded, current):
                break

            current = eroded
            loop_idx += 1

        return paths

    def _ensure_rgb01(
        self,
        image_source: Union[str, np.ndarray]
    ) -> np.ndarray:
        """Hilfsfunktion: lädt/konvertiert ein Bild in RGB float32 [0,1]."""

        if isinstance(image_source, str):
            img_bgr = cv2.imread(image_source, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Konnte Bild nicht laden: {image_source}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return (img_rgb.astype(np.float32) / 255.0).clip(0, 1)

        img_arr = np.asarray(image_source)
        if img_arr.ndim != 3 or img_arr.shape[2] != 3:
            raise ValueError("expect image as (H,W,3) array")

        img_arr = img_arr.astype(np.float32)
        if img_arr.max() > 1.0:
            img_arr /= 255.0
        return img_arr.clip(0, 1)

    def extract_color_layers(
        self,
        image_source: Union[str, np.ndarray],
        *,
        k_colors: Optional[int] = None,
        k_min: int = 8,
        k_max: int = 16,
        use_dither: bool = True,
        min_path_length: int = 2,
        min_area_ratio: float = 0.0005,
        stroke_spacing_scale: float = 1.0,
        preserve_edge_strokes: bool = False,
        detail_edge_boost: float = 1.0,
        edge_sensitivity: float = 1.0,
        microtransition_boost: float = 1.0,
        chroma_boost: float = 1.0,
        highlight_boost: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Analysiert ein Bild (Dateipfad oder RGB-Array) und erzeugt
        farbbasierte Layer via adaptiver K-Means-Quantisierung in CIELAB.

        Rückgabe:
            Liste von Layer-Dicts mit
              "color_rgb": (r,g,b) in 0..255
              "pixel_paths": Liste von Pfaden ([(x_px, y_px), ...])
              "label": Cluster-Index

        Zusätzlich wird das letzte Analyse-Ergebnis in
        ``self.last_color_analysis`` abgelegt (z.B. für Preview-Zwecke).

        Parameter wie ``stroke_spacing_scale`` und ``detail_edge_boost`` ermöglichen
        es, deutlich feinere Schraffuren sowie zusätzliche Kantenspuren zu erzeugen,
        wodurch einzelne Pinselstriche besser nachvollzogen werden können. Über
        ``microtransition_boost``, ``chroma_boost`` und ``highlight_boost`` lassen
        sich außerdem Übergänge verdichten, Farbakzente verstärken und Lichter
        gezielt anheben.
        """

        enhanced_bgr = self.enhance_image_quality(image_source)
        if enhanced_bgr is not None and self.last_enhanced_rgb01 is not None:
            img_srgb01 = self.last_enhanced_rgb01.copy()
        else:
            img_srgb01 = self._ensure_rgb01(image_source)
        detail_scale = float(np.clip(detail_edge_boost, 0.3, 4.0))
        spacing_scale = float(np.clip(stroke_spacing_scale, 0.2, 2.5))
        edge_sense = float(np.clip(edge_sensitivity, 0.25, 4.0))
        min_area_ratio = float(np.clip(min_area_ratio, 1e-6, 0.01))
        micro_boost = float(np.clip(microtransition_boost, 0.5, 3.0))
        chroma_boost = float(np.clip(chroma_boost, 0.5, 2.5))
        highlight_boost = float(np.clip(highlight_boost, 0.0, 1.5))

        detail_scale *= micro_boost

        img_u8 = (img_srgb01.clip(0, 1) * 255).astype(np.uint8)
        img_gray_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

        orig_h, orig_w = img_srgb01.shape[:2]

        # Hohe Auflösung bremst KMeans stark aus. Um UI-Timeouts zu vermeiden,
        # rechnen wir auf einer verkleinerten Kopie (max_dim Pixel) und
        # skalieren die Ergebnisse anschließend wieder auf die Originalgröße.
        max_dim = int(800 * float(np.clip(np.sqrt(detail_scale), 1.0, 2.1)))
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            img_srgb01_proc = cv2.resize(
                img_srgb01,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img_srgb01_proc = img_srgb01

        if k_colors is not None:
            k_min = max(1, int(k_colors))
            k_max = k_min

        img8 = (img_srgb01_proc.clip(0, 1) * 255).astype(np.uint8)
        lab_cv = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_cv)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        lab_cv = cv2.merge([l, a, b])
        rgb_cv = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB)
        smooth = cv2.bilateralFilter(rgb_cv, d=9, sigmaColor=75, sigmaSpace=75)
        pre_rgb01 = (smooth.astype(np.float32) / 255.0).clip(0, 1)

        if chroma_boost != 1.0 or highlight_boost > 0.0:
            hsv = cv2.cvtColor(pre_rgb01.astype(np.float32), cv2.COLOR_RGB2HSV)
            if chroma_boost != 1.0:
                hsv[..., 1] = np.clip(hsv[..., 1] * chroma_boost, 0.0, 1.0)
            if highlight_boost > 0.0:
                v = hsv[..., 2]
                v = np.clip(v + highlight_boost * np.power(v, 2.2), 0.0, 1.0)
                hsv[..., 2] = v
            pre_rgb01 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).clip(0.0, 1.0)

        lab = rgb2lab(pre_rgb01)
        H, W, _ = lab.shape

        base_segments = (H * W) / 1800
        target_segments = int(np.clip(base_segments * detail_scale, 220, 3600))
        slic_compactness = float(np.clip(12.0 / max(detail_scale, 0.5), 5.0, 18.0))
        segments = slic(
            pre_rgb01,
            n_segments=target_segments,
            compactness=slic_compactness,
            sigma=1,
            start_label=0,
        ).astype(np.int32)

        num_segments = int(segments.max()) + 1
        seg_flat = segments.reshape(-1)
        lab_flat = lab.reshape(-1, 3)
        counts = np.bincount(seg_flat, minlength=num_segments).astype(np.float32)
        mean_lab = np.zeros((num_segments, 3), dtype=np.float32)

        valid_mask = counts > 0
        valid_indices = np.where(valid_mask)[0]

        if valid_indices.size == 0:
            return []

        for channel in range(3):
            sums = np.bincount(
                seg_flat,
                weights=lab_flat[:, channel],
                minlength=num_segments,
            )
            mean_lab[:, channel] = sums / np.maximum(counts, 1e-6)

        mean_lab_valid = mean_lab[valid_indices]

        k_auto_estimate = (np.sqrt(H * W) / 300.0) * max(detail_scale, 0.6)
        k_auto = int(np.clip(int(round(k_auto_estimate)), k_min, k_max))
        k_auto = max(1, min(k_auto, mean_lab_valid.shape[0]))

        km = _fit_kmeans_compat(
            mean_lab_valid,
            n_clusters=k_auto,
            random_state=0,
        )
        centers_lab = km.cluster_centers_

        superpixel_labels = np.zeros(num_segments, dtype=np.int32)
        superpixel_labels[valid_indices] = km.labels_
        labels = superpixel_labels[segments]

        if use_dither and centers_lab.shape[0] < 256:
            labels = cv2.medianBlur(labels.astype(np.uint8), 3).astype(np.int32)

        lab_used = centers_lab[labels]

        palette_rgb01 = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3).clip(0, 1)
        quant_rgb01 = lab2rgb(lab_used).clip(0, 1)

        # Falls wir verkleinert haben, Ergebnisse wieder auf Originalgröße bringen.
        if (H, W) != (orig_h, orig_w):
            labels_full = cv2.resize(
                labels.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
            quant_rgb01 = cv2.resize(
                quant_rgb01,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            pre_rgb01 = cv2.resize(
                pre_rgb01,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            labels_full = labels

        # Analyse-Ergebnis für Debug/Preview merken
        self.last_color_analysis = {
            "preprocessed_rgb01": pre_rgb01,
            "labels": labels_full.astype(np.int32),
            "centers_lab": centers_lab,
            "palette_rgb01": palette_rgb01,
            "quant_rgb01": quant_rgb01.astype(np.float32),
        }

        layers: List[Dict[str, Any]] = []
        unique_labels = np.unique(labels_full)

        kernel_size = int(np.clip(round(3 / max(detail_scale, 0.5)), 1, 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        min_area = max(int(min_area_ratio * orig_h * orig_w), 32)

        def _remove_small_regions(mask_arr: np.ndarray, min_pixels: int) -> np.ndarray:
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
                mask_arr,
                connectivity=8,
            )
            cleaned_mask = np.zeros_like(mask_arr)
            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area >= min_pixels:
                    cleaned_mask[labels_im == comp_id] = 255
            return cleaned_mask

        blur_kernel = int(np.clip(3 + (1 if detail_scale < 1.0 else 0), 3, 7))
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred_gray = cv2.GaussianBlur(img_gray_u8, (blur_kernel, blur_kernel), 0)

        for li in unique_labels:
            mask = (labels_full == li).astype(np.uint8) * 255

            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if cleaned.max() == 0:
                cleaned = mask

            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            cleaned = _remove_small_regions(cleaned, min_area)
            if cleaned.max() == 0:
                cleaned = mask

            base_spacing = max(1, int(round(min(orig_h, orig_w) / 180)))
            spacing_px = int(np.clip(round(base_spacing * spacing_scale), 1, 15))

            paths = self.mask_to_offset_contours(
                cleaned,
                spacing_px=spacing_px,
                min_len=min_path_length,
            )
            if not paths:
                paths = self.extract_paths_from_mask(cleaned)
            filtered_paths = [
                path for path in paths if len(path) >= min_path_length
            ]

            if preserve_edge_strokes:
                low_thr = int(np.clip(32 / edge_sense, 6, 120))
                high_thr = int(np.clip(140 / edge_sense, low_thr + 10, 255))
                edge_mask = cv2.Canny(blurred_gray, low_thr, high_thr)
                edge_mask = cv2.bitwise_and(edge_mask, cleaned)
                if detail_scale > 1.0:
                    extra_iter = int(np.clip(round(detail_scale - 1.0), 1, 3))
                    edge_mask = cv2.dilate(
                        edge_mask,
                        np.ones((3, 3), np.uint8),
                        iterations=extra_iter,
                    )
                edge_paths = self.extract_paths_from_mask(edge_mask)
                for path in edge_paths:
                    if len(path) >= 2:
                        filtered_paths.append(path)

            color_rgb = tuple(
                int(np.clip(round(c * 255), 0, 255)) for c in palette_rgb01[int(li)]
            )

            layers.append({
                "color_rgb": color_rgb,
                "pixel_paths": filtered_paths,
                "label": int(li),
                "_lab_l": float(centers_lab[int(li)][0]),
            })

        layers = sorted(layers, key=lambda entry: entry["_lab_l"])
        for layer in layers:
            layer.pop("_lab_l", None)

        return layers


def preprocess_for_slicing(img_srgb01: np.ndarray) -> np.ndarray:
    """Bilateral + CLAHE für bessere Tonwerte ohne Kantenverlust."""
    img8 = (img_srgb01 * 255).astype(np.uint8)
    lab = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    lab = cv2.merge([l, a, b])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    smooth = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    return (smooth.astype(np.float32) / 255.0).clip(0, 1)


def quantize_adaptive_lab(img_srgb01: np.ndarray, k_min: int = 8, k_max: int = 16):
    """Adaptive Farbquantisierung in LAB + leichter FS-Error-Diffusion."""
    lab = rgb2lab(img_srgb01)
    H, W, _ = lab.shape
    X = lab.reshape(-1, 3)

    k = int(np.clip(int(np.sqrt(H * W) / 300), k_min, k_max))
    km = _fit_kmeans_compat(
        X,
        n_clusters=k,
        random_state=0,
    )
    centers = km.cluster_centers_
    labels = km.labels_.reshape(H, W)

    lab_q = lab.copy()
    for y in range(H - 1):
        for x in range(1, W - 1):
            old = lab_q[y, x]
            idx = np.argmin(np.linalg.norm(centers - old, axis=1))
            new = centers[idx]
            err = old - new
            lab_q[y, x] = new
            lab_q[y, x + 1] += err * 7 / 16
            lab_q[y + 1, x - 1] += err * 3 / 16
            lab_q[y + 1, x] += err * 5 / 16
            lab_q[y + 1, x + 1] += err * 1 / 16

    return lab_q, centers, labels


def extract_edges(img_gray01: np.ndarray):
    """Kanten für Detail-Striche (fine_brush)."""
    g8 = (img_gray01 * 255).astype(np.uint8)
    edges = cv2.Canny(g8, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines = [c[:, 0, :] for c in cnts if len(c) > 10]
    return polylines

