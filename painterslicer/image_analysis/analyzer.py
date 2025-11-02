import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Union
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb


class ImageAnalyzer:
    """Bildanalyse / Vorverarbeitung."""

    def __init__(self):
        self.last_color_analysis: Optional[Dict[str, Any]] = None

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
        img = self.load_image(image_path)
        mask = self.edge_mask(img)
        return mask

    # ----------------------------
    #  NEU: Layer-Extraktion
    # ----------------------------
    def make_layer_masks(self, image_path: str) -> Dict[str, Optional[np.ndarray]]:
        """
        Gibt drei 'Malschichten'-Masken zurück:
        - background_mask: wenig Detail (grobe Flächen)
        - mid_mask: mittleres Detail
        - detail_mask: feine/high-contrast Kanten

        Rückgabe: dict mit drei 2D-Numpy-Arrays (uint8)
        """

        img = self.load_image(image_path)
        edges = self.edge_mask(img)  # 0..255, weiße Kanten auf schwarz

        if edges is None:
            return {
                "background_mask": None,
                "mid_mask": None,
                "detail_mask": None,
            }

        # Kanten sind 0 (schwarz) oder 255 (weiß).
        # Wir glätten das und schauen uns "wo häuft sich Struktur" an.

        # 1) Kantendichte aufweichen -> lokale Strukturintensität
        blur_edges = cv2.GaussianBlur(edges, (21, 21), 0)

        # blur_edges: helle Bereiche = dort gibt es viele Linien / Details

        # 2) Schwellwerte setzen
        #    (Werte hier sind heuristisch, kannst du später tweaken)
        high_thresh = np.percentile(blur_edges, 85)  # sehr detailreich
        mid_thresh = np.percentile(blur_edges, 50)   # mittlere Struktur

        # detail_mask: alles was sehr viele Details hat
        detail_mask = (blur_edges >= high_thresh).astype(np.uint8) * 255

        # mid_mask: mittlere Struktur, aber nicht schon als "detail" markiert
        mid_mask = ((blur_edges >= mid_thresh) & (blur_edges < high_thresh)).astype(np.uint8) * 255

        # background_mask: der Rest
        background_mask = (blur_edges < mid_thresh).astype(np.uint8) * 255

        # Optional ein bisschen schließen / füllen für Hintergrund, damit das schön großflächig wird:
        kernel = np.ones((7, 7), np.uint8)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)

        return {
            "background_mask": background_mask,
            "mid_mask": mid_mask,
            "detail_mask": detail_mask,
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
        """

        img_srgb01 = self._ensure_rgb01(image_source)

        if k_colors is not None:
            k_min = max(1, int(k_colors))
            k_max = k_min

        img8 = (img_srgb01.clip(0, 1) * 255).astype(np.uint8)
        lab_cv = cv2.cvtColor(img8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_cv)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        lab_cv = cv2.merge([l, a, b])
        rgb_cv = cv2.cvtColor(lab_cv, cv2.COLOR_LAB2RGB)
        smooth = cv2.bilateralFilter(rgb_cv, d=9, sigmaColor=75, sigmaSpace=75)
        pre_rgb01 = (smooth.astype(np.float32) / 255.0).clip(0, 1)

        lab = rgb2lab(pre_rgb01)
        H, W, _ = lab.shape
        X = lab.reshape(-1, 3)

        k_auto = int(np.clip(int(np.sqrt(H * W) / 300), k_min, k_max))
        km = KMeans(n_clusters=k_auto, n_init="auto", random_state=0).fit(X)
        centers_lab = km.cluster_centers_
        labels = km.labels_.reshape(H, W)

        if use_dither:
            lab_q = lab.copy()
            for y in range(H - 1):
                for x in range(1, W - 1):
                    old = lab_q[y, x]
                    idx = int(np.argmin(np.linalg.norm(centers_lab - old, axis=1)))
                    new = centers_lab[idx]
                    err = old - new
                    lab_q[y, x] = new
                    lab_q[y, x + 1]     += err * (7 / 16)
                    lab_q[y + 1, x - 1] += err * (3 / 16)
                    lab_q[y + 1, x]     += err * (5 / 16)
                    lab_q[y + 1, x + 1] += err * (1 / 16)

            Xq = lab_q.reshape(-1, 3)
            dists = np.linalg.norm(
                Xq[:, None, :] - centers_lab[None, :, :],
                axis=2,
            )
            labels = np.argmin(dists, axis=1).reshape(H, W)
            lab_used = lab_q
        else:
            lab_used = centers_lab[labels]

        palette_rgb01 = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3).clip(0, 1)
        quant_rgb01 = lab2rgb(lab_used).clip(0, 1)

        # Analyse-Ergebnis für Debug/Preview merken
        self.last_color_analysis = {
            "preprocessed_rgb01": pre_rgb01,
            "labels": labels.astype(np.int32),
            "centers_lab": centers_lab,
            "palette_rgb01": palette_rgb01,
            "quant_rgb01": quant_rgb01.astype(np.float32),
        }

        layers: List[Dict[str, Any]] = []
        unique_labels = np.unique(labels)

        kernel = np.ones((3, 3), np.uint8)

        for li in unique_labels:
            mask = (labels == li).astype(np.uint8) * 255

            # Kleine Artefakte entfernen
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if cleaned.max() == 0:
                cleaned = mask

            paths = self.extract_paths_from_mask(cleaned)
            filtered_paths = [
                path for path in paths if len(path) >= min_path_length
            ]

            color_rgb = tuple(
                int(np.clip(round(c * 255), 0, 255)) for c in palette_rgb01[int(li)]
            )

            layers.append({
                "color_rgb": color_rgb,
                "pixel_paths": filtered_paths,
                "label": int(li),
            })

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
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
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
