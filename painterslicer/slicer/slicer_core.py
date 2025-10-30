from typing import Dict, Any, List, Tuple
import cv2
from config.config_loader import load_machine_config, load_brush_presets
import numpy as np
from skimage.color import lab2rgb
from utils.color_model import PaintColorModel


class PainterSlicer:
    """
    Übersetzt Analyse-Ergebnisse in:
    - logische Mal-Reihenfolge
    - normalisierte Strichpfade (mm)
    - PaintCode mit Z und Pressure
    """

    def __init__(self):
        # maschinenparameter aus config
        self.machine_cfg = load_machine_config()
        self.work_w_mm = float(self.machine_cfg["work_area_mm"]["x"])
        self.work_h_mm = float(self.machine_cfg["work_area_mm"]["y"])

        # slicer/slicer_core.py (im SlicerCore oder deiner Slicer-Klasse)
        self.grid_mm = 0.3  # vorher evtl. 1.0; feiner = mehr Details
        self.num_glaze_passes = 3  # 2..3 bringt sichtbare Glätte

        # Z-Default: benutze home_position.z als Z_UP, paint_z_mm als Z_DOWN-Position (wenn gesetzt)
        self.z_up_default = float(self.machine_cfg.get("home_position", {}).get("z", 50.0))
        self.z_down_default = float(self.machine_cfg.get("paint_z_mm", 5.0))

        # brush presets
        self.brush_presets = load_brush_presets()

    def generate_paint_plan(self, layer_masks: Dict[str, Any]) -> dict:
        plan = [
            {
                "layer": "background",
                "tool": "broad_brush",
                "comment": "Große Flächen mit wenig Details zuerst.",
                "mask_present": layer_masks.get("background_mask") is not None
            },
            {
                "layer": "mid",
                "tool": "medium_brush",
                "comment": "Struktur und Volumen.",
                "mask_present": layer_masks.get("mid_mask") is not None
            },
            {
                "layer": "detail",
                "tool": "fine_brush",
                "comment": "Feine Konturen, Highlights, scharfe Kanten zuletzt.",
                "mask_present": layer_masks.get("detail_mask") is not None
            }
        ]

        return {"steps": plan}

    def normalize_paths_to_mm(
        self,
        image_path: str,
        pixel_paths: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[float, float]]]:
        img = cv2.imread(image_path)
        if img is None:
            return []

        img_h, img_w = img.shape[:2]

        norm_paths_mm: List[List[Tuple[float, float]]] = []
        for path in pixel_paths:
            mm_path: List[Tuple[float, float]] = []
            for (x_px, y_px) in path:
                X_mm = (x_px / img_w) * self.work_w_mm
                Y_mm = (y_px / img_h) * self.work_h_mm
                mm_path.append((X_mm, Y_mm))
            norm_paths_mm.append(mm_path)

        return norm_paths_mm

    def _get_brush_pressure(self, tool_name: str) -> float:
        """
        Hole empfohlenen Druck (0..1) aus brush_presets.json.
        Falls nicht vorhanden, Standard 0.3 zurückgeben.
        """
        preset = self.brush_presets.get(tool_name, {})
        val = preset.get("max_pressure", None)
        try:
            return float(val) if val is not None else 0.3
        except Exception:
            return 0.3

    def paths_to_paintcode(
        self,
        mm_paths: List[List[Tuple[float, float]]],
        *,
        tool_name: str,
        pressure: float,
        z_up: float,
        z_down: float
    ) -> str:
        """
        Baut PaintCode mit Werkzeugwahl, Druck, Z-Up/Down, Farbaufnahme und optional Reinigung.
        """

        lines = []
        stroke_id = 1

        # Hole Brush-Metadaten aus Presets (Farbe, Cleaning)
        brush_meta = self.brush_presets.get(tool_name, {})
        color_rgb = brush_meta.get("color_rgb", [0, 0, 0])
        needs_cleaning = bool(brush_meta.get("needs_cleaning", True))

        # Station-Koordinaten (Palette / Wash)
        pal = self.machine_cfg.get("palette_station", {"x": 0, "y": 0, "z": z_down})
        wash = self.machine_cfg.get("wash_station", {"x": 0, "y": 0, "z": z_up})

        pal_x = float(pal.get("x", 0.0))
        pal_y = float(pal.get("y", 0.0))
        pal_z = float(pal.get("z", z_down))

        wash_x = float(wash.get("x", 0.0))
        wash_y = float(wash.get("y", 0.0))
        wash_z = float(wash.get("z", z_up))

        CLEAN_INTERVAL = 5  # z.B. nach jedem 5. Stroke sauber machen

        for path in mm_paths:
            if len(path) < 1:
                continue

            start_x, start_y = path[0]

            lines.append(f"; Stroke {stroke_id}")
            lines.append(f"TOOL {tool_name}")
            lines.append(f"PRESSURE {pressure:.2f}")

            # --- Farbaufnahme an der Palette ---
            lines.append(
                f"GOTO_PALETTE X{pal_x:.2f} Y{pal_y:.2f} Z{z_up:.2f}"
            )
            lines.append(
                f"LOAD_COLOR R{color_rgb[0]} G{color_rgb[1]} B{color_rgb[2]}"
            )
            # optional: hier könnte man DIP / STIR / etc. simulieren
            lines.append(
                f"DIP_TOOL X{pal_x:.2f} Y{pal_y:.2f} Z{pal_z:.2f}"
            )

            # --- Anfahrt Stroke-Start ---
            lines.append(
                f"Z_UP X{start_x:.2f} Y{start_y:.2f} Z{z_up:.2f}"
            )
            lines.append(
                f"Z_DOWN X{start_x:.2f} Y{start_y:.2f} Z{z_down:.2f}"
            )

            # Pinsel aufsetzen
            lines.append(
                f"PEN_DOWN X{start_x:.2f} Y{start_y:.2f}"
            )

            # Stroke abfahren
            for (x_mm, y_mm) in path[1:]:
                lines.append(f"MOVE X{x_mm:.2f} Y{y_mm:.2f}")

            # Pinsel lösen
            lines.append("PEN_UP")

            # Hub hochfahren am Endpunkt
            end_x, end_y = path[-1]
            lines.append(
                f"Z_UP X{end_x:.2f} Y{end_y:.2f} Z{z_up:.2f}"
            )

            # --- Optional Reinigung nach bestimmten Strokes ---
            if needs_cleaning and (stroke_id % CLEAN_INTERVAL == 0):
                lines.append(
                    f"GOTO_WASH X{wash_x:.2f} Y{wash_y:.2f} Z{z_up:.2f}"
                )
                lines.append(
                    f"CLEAN_TOOL X{wash_x:.2f} Y{wash_y:.2f} Z{wash_z:.2f}"
                )

            lines.append("")  # Leerzeile zur Lesbarkeit
            stroke_id += 1

        if not lines:
            return "; (keine Pfade gefunden)"

        return "\n".join(lines)



    def normalize_color_layers_to_mm(self, image_path: str, color_layers: list):
        """
        Nimmt die color_layers vom Analyzer (jede hat pixel_paths + color_rgb)
        und rechnet alle pixel_paths in mm um.

        Rückgabe:
        [
            {
                "color_rgb": (r,g,b),
                "mm_paths": [ [(x_mm,y_mm), (x_mm,y_mm), ...], [...], ... ]
            },
            ...
        ]
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        img_h, img_w = img.shape[:2]

        norm_layers = []

        for layer in color_layers:
            pixel_paths = layer["pixel_paths"]
            rgb = layer["color_rgb"]

            mm_paths = []
            for path in pixel_paths:
                mm_path = []
                for (x_px, y_px) in path:
                    X_mm = (x_px / img_w) * self.work_w_mm
                    Y_mm = (y_px / img_h) * self.work_h_mm
                    mm_path.append((X_mm, Y_mm))
                if len(mm_path) > 1:
                    mm_paths.append(mm_path)

            norm_layers.append({
                "color_rgb": rgb,
                "mm_paths": mm_paths
            })

        return norm_layers





    def multi_layer_paintcode(
        self,
        normalized_layers: list,
        *,
        tool_name: str,
        pressure: float,
        z_up: float,
        z_down: float,
        clean_interval: int = 5
    ) -> str:
        """
        Erzeugt PaintCode für MEHRERE Farblayer.
        Jeder Layer hat:
          - eigene Farbe (color_rgb)
          - eigene mm_paths
        Wir fahren:
          1. Farbaufnahme mit LOAD_COLOR R G B
          2. Pfade malen
          3. Nach clean_interval Strokes -> Waschstation
        """

        lines = []
        stroke_global_id = 1

        # hole Stationsdaten
        brush_meta = self.brush_presets.get(tool_name, {})
        needs_cleaning = bool(brush_meta.get("needs_cleaning", True))

        pal = self.machine_cfg.get("palette_station", {"x": 0, "y": 0, "z": z_down})
        wash = self.machine_cfg.get("wash_station", {"x": 0, "y": 0, "z": z_up})

        pal_x = float(pal.get("x", 0.0))
        pal_y = float(pal.get("y", 0.0))
        pal_z = float(pal.get("z", z_down))

        wash_x = float(wash.get("x", 0.0))
        wash_y = float(wash.get("y", 0.0))
        wash_z = float(wash.get("z", z_up))

        # --- Farbreihenfolge anpassen (hell -> dunkel) ---
        def luminance(rgb):
            r, g, b = rgb
            return 0.2126*r + 0.7152*g + 0.0722*b

        normalized_layers = sorted(
            normalized_layers,
            key=lambda L: luminance(L["color_rgb"]),
            reverse=False  # False = hell zuerst
        )


        for layer_idx, layer in enumerate(normalized_layers):
            color_rgb = layer["color_rgb"]
            mm_paths = layer["mm_paths"]

            if not mm_paths:
                continue

            lines.append(f"; ===============================")
            lines.append(f"; Farb-Layer {layer_idx+1}  RGB={color_rgb}")
            lines.append(f"; TOOL {tool_name}")
            lines.append(f"; ===============================")

            # vor JEDEM Farb-Layer: zur Palette, Farbe laden
            lines.append(
                f"GOTO_PALETTE X{pal_x:.2f} Y{pal_y:.2f} Z{z_up:.2f}"
            )
            lines.append(
                f"LOAD_COLOR R{color_rgb[0]} G{color_rgb[1]} B{color_rgb[2]}"
            )
            lines.append(
                f"DIP_TOOL X{pal_x:.2f} Y{pal_y:.2f} Z{pal_z:.2f}"
            )
            lines.append(f"PRESSURE {pressure:.2f}")
            lines.append(f"TOOL {tool_name}")
            lines.append("")

            local_count_since_clean = 0

            for path in mm_paths:
                if len(path) < 2:
                    continue

                start_x, start_y = path[0]

                lines.append(f"; Stroke {stroke_global_id}")
                lines.append(
                    f"Z_UP X{start_x:.2f} Y{start_y:.2f} Z{z_up:.2f}"
                )
                lines.append(
                    f"Z_DOWN X{start_x:.2f} Y{start_y:.2f} Z{z_down:.2f}"
                )
                lines.append(
                    f"PEN_DOWN X{start_x:.2f} Y{start_y:.2f}"
                )

                for (x_mm, y_mm) in path[1:]:
                    lines.append(f"MOVE X{x_mm:.2f} Y{y_mm:.2f}")

                lines.append("PEN_UP")

                end_x, end_y = path[-1]
                lines.append(
                    f"Z_UP X{end_x:.2f} Y{end_y:.2f} Z{z_up:.2f}"
                )
                lines.append("")

                stroke_global_id += 1
                local_count_since_clean += 1

                # Reinigung?
                if needs_cleaning and local_count_since_clean >= clean_interval:
                    lines.append(
                        f"GOTO_WASH X{wash_x:.2f} Y{wash_y:.2f} Z{z_up:.2f}"
                    )
                    lines.append(
                        f"CLEAN_TOOL X{wash_x:.2f} Y{wash_y:.2f} Z{wash_z:.2f}"
                    )
                    lines.append("")
                    local_count_since_clean = 0

            # nach Layer optional reinigen
            if needs_cleaning:
                lines.append(
                    f"GOTO_WASH X{wash_x:.2f} Y{wash_y:.2f} Z{z_up:.2f}"
                )
                lines.append(
                    f"CLEAN_TOOL X{wash_x:.2f} Y{wash_y:.2f} Z{wash_z:.2f}"
                )
                lines.append("")

        if not lines:
            return "; (keine malbaren Layer gefunden)"

        return "\n".join(lines)




def _luminance_lab(lab_color):
    # Helligkeit aus L*
    return float(lab_color[0])

def labels_to_layer_paths(labels: np.ndarray, centers_lab: np.ndarray, work_w_mm: float, work_h_mm: float, grid_mm: float):
    """
    Sehr einfache Flächen-Strokes: wir rasterisieren pro Label in Linien.
    Später: SLIC/Schraffur/Gradientenrichtung verfeinern.
    """
    H, W = labels.shape
    sx = max(1, int((grid_mm / work_w_mm) * W))
    sy = max(1, int((grid_mm / work_h_mm) * H))

    layers = []
    uniq = np.unique(labels)
    for li in uniq:
        mask = (labels == li).astype(np.uint8)
        paths = []
        # naive Zeilen-Füllung (besser: Boustrophedon & Rand-Offset)
        for y in range(0, H, sy):
            xs = np.where(mask[y] > 0)[0]
            if xs.size == 0:
                continue
            # zusammenhängende Segmente zu Linien konvertieren
            start = None
            for x in range(0, W, sx):
                if mask[y, x] > 0 and start is None:
                    start = x
                if (start is not None) and (mask[y, x] == 0 or x+sx >= W):
                    end = x if mask[y,x]==0 else min(W-1, x+sx)
                    # Linie in mm
                    x1_mm = (start / W) * work_w_mm
                    y1_mm = (y     / H) * work_h_mm
                    x2_mm = (end   / W) * work_w_mm
                    y2_mm = y1_mm
                    paths.append([(x1_mm,y1_mm),(x2_mm,y2_mm)])
                    start = None

        layers.append({
            "label": int(li),
            "lab": centers_lab[li],
            "paths_mm": paths
        })
    return layers

def sort_layers_light_to_dark(layers):
    return sorted(layers, key=lambda L: _luminance_lab(L["lab"]))


def build_paintcode_from_layers(layers, color_model: PaintColorModel):
    """
    Erzeugt Stroke-Timeline: hell -> dunkel, pro Layer mehrere 'glaze' Pässe.
    Rückgabe: Liste von {'color_rgb': (r,g,b), 'points': [(x,y)...], 'tool': '...'}
    """
    timeline = []
    # hell -> dunkel
    for L in sort_layers_light_to_dark(layers):
        lab = L["lab"]
        rgb01 = lab2rgb(lab.reshape(1,1,3)).reshape(3,)
        # Maschinen-Param aus Farbmodell (hier Dummy)
        _params = color_model.predict_params_from_srgb01(rgb01.reshape(1,3))[0]

        # Tool-Heuristik (später: aus _params ableiten)
        tool = "medium_brush"
        color_rgb = tuple(int(255*c) for c in rgb01.clip(0,1))

        for _pass in range(getattr(self, "num_glaze_passes", 2)):
            for path in L["paths_mm"]:
                if len(path) < 2:
                    continue
                timeline.append({
                    "color_rgb": color_rgb,
                    "points": path,
                    "tool": tool
                })
    # Beispiel innerhalb deiner Slicer-Methode:
    # lab_q, centers, labels hast du aus analyzer.quantize_adaptive_lab(...)
    layers = labels_to_layer_paths(labels, centers, self.work_w_mm, self.work_h_mm, self.grid_mm)

    # Farbmodell initialisieren (später: fit mit echten Kalibrierdaten)
    pcm = PaintColorModel()
    pcm.fit_dummy()

    timeline = build_paintcode_from_layers(layers, pcm)

    # -> an UI geben:
    return timeline
