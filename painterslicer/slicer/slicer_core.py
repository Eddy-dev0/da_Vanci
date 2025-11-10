from typing import Dict, Any, List, Tuple, Optional, Iterable
import cv2
from painterslicer.config.config_loader import load_machine_config, load_brush_presets
import numpy as np
from skimage.color import lab2rgb
from painterslicer.utils.color_model import PaintColorModel
from .layers import PaintLayer


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
        self.clean_interval_default = 5

        # Z-Default: benutze home_position.z als Z_UP, paint_z_mm als Z_DOWN-Position (wenn gesetzt)
        self.z_up_default = float(self.machine_cfg.get("home_position", {}).get("z", 50.0))
        self.z_down_default = float(self.machine_cfg.get("paint_z_mm", 5.0))

        # brush presets
        self.brush_presets = load_brush_presets()
        self._brush_overrides: Dict[str, Dict[str, Any]] = {}

    def apply_style_profile(self, profile: Optional[Dict[str, Any]]):
        """Übernimmt Stil-Parameter für den Slicer (Raster, Glaze-Pässe, Reinigung)."""

        if not profile:
            return

        grid_mm = profile.get("grid_mm")
        if grid_mm is not None:
            try:
                self.grid_mm = float(grid_mm)
            except (TypeError, ValueError):
                pass

        glaze_passes = profile.get("num_glaze_passes")
        if glaze_passes is not None:
            try:
                self.num_glaze_passes = max(1, int(glaze_passes))
            except (TypeError, ValueError):
                pass

        clean_interval = profile.get("clean_interval")
        if clean_interval is not None:
            try:
                self.clean_interval_default = max(1, int(clean_interval))
            except (TypeError, ValueError):
                pass

    def apply_brush_overrides(self, overrides: Optional[Dict[str, Dict[str, Any]]]) -> None:
        self._brush_overrides = {k: dict(v) for k, v in (overrides or {}).items()}

    def get_brush_settings(self, tool_name: str) -> Dict[str, Any]:
        base = dict(self.brush_presets.get(tool_name, {}))
        override = self._brush_overrides.get(tool_name, {})
        base.update({k: v for k, v in override.items() if v is not None})
        return base

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
        preset = self.get_brush_settings(tool_name)
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
        brush_meta = self.get_brush_settings(tool_name)
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



    def normalize_color_layers_to_mm(
        self,
        image_path: str,
        color_layers: list,
        *,
        style_key: Optional[str] = None,
    ) -> List[PaintLayer]:
        """
        Nimmt die ``color_layers`` vom Analyzer (jede hat ``pixel_paths`` + reichhaltige
        Metadaten) und rechnet alle ``pixel_paths`` in mm um. Zusätzlich werden die
        Analyseattribute (Stage, Werkzeugempfehlung, Kennzahlen …) konserviert, damit der
        Slice-Prozess diese Informationen vollständig ausnutzen kann.

        Rückgabe:
        List[:class:`PaintLayer`] preserving all analysis metadata while exposing
        millimetre converted paths, inferred depth ordering, opacity and brush
        configuration for every layer.
        """

        img = cv2.imread(image_path)
        if img is None:
            return []

        img_h, img_w = img.shape[:2]

        # Reihenfolge beibehalten: zuerst nach explizitem Order, ansonsten nach Stage
        stage_priority = {"background": 0, "mid": 1, "detail": 2}

        indexed_layers = list(enumerate(color_layers))
        indexed_layers.sort(
            key=lambda entry: (
                entry[1].get("order", 1_000_000 + entry[0]),
                stage_priority.get(entry[1].get("stage"), 1),
                -float(entry[1].get("coverage", 0.0)),
            )
        )

        norm_layers: List[PaintLayer] = []

        for idx, layer in indexed_layers:
            pixel_paths = layer.get("pixel_paths", []) or []
            rgb = tuple(layer.get("color_rgb", (0, 0, 0)))

            mm_paths: List[List[Tuple[float, float]]] = []
            for path in pixel_paths:
                mm_path = []
                for (x_px, y_px) in path:
                    X_mm = (x_px / img_w) * self.work_w_mm
                    Y_mm = (y_px / img_h) * self.work_h_mm
                    mm_path.append((X_mm, Y_mm))
                if len(mm_path) > 1:
                    mm_paths.append(mm_path)

            # Helligkeitsheuristik für Fallback-Sortierungen in späteren Schritten
            luminance = (
                0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                if rgb
                else 0.0
            )

            tool_name = layer.get("tool")
            brush_meta = self.get_brush_settings(tool_name) if tool_name else {}
            depth = float(layer.get("depth", 0.0))
            if not depth:
                stage = layer.get("stage")
                base_stage = stage_priority.get(stage, 1)
                depth = base_stage * 1000.0 + float(layer.get("order", idx))
                if layer.get("shading") == "highlight":
                    depth += 0.1
                elif layer.get("shading") == "shadow":
                    depth += 0.2

            opacity = float(brush_meta.get("opacity", 1.0)) if brush_meta else 1.0
            mask = layer.get("mask")

            paint_layer = PaintLayer(
                color_rgb=rgb,
                mm_paths=mm_paths,
                stage=layer.get("stage"),
                technique=layer.get("technique"),
                shading=layer.get("shading"),
                coverage=float(layer.get("coverage", 0.0)),
                order=int(layer.get("order", idx)),
                label=layer.get("label"),
                depth=depth,
                opacity=opacity,
                style_key=style_key,
                mask=mask,
                brush=dict(brush_meta),
                tool=tool_name,
                metadata={
                    "detail_ratio": float(layer.get("detail_ratio", 0.0)),
                    "mid_ratio": float(layer.get("mid_ratio", 0.0)),
                    "background_ratio": float(layer.get("background_ratio", 0.0)),
                    "detail_strength": float(layer.get("detail_strength", 0.0)),
                    "mid_strength": float(layer.get("mid_strength", 0.0)),
                    "background_strength": float(layer.get("background_strength", 0.0)),
                    "texture_strength": float(layer.get("texture_strength", 0.0)),
                    "highlight_strength": float(layer.get("highlight_strength", 0.0)),
                    "shadow_strength": float(layer.get("shadow_strength", 0.0)),
                    "contrast_strength": float(layer.get("contrast_strength", 0.0)),
                    "color_variance_strength": float(layer.get("color_variance_strength", 0.0)),
                    "path_count": int(layer.get("path_count", len(mm_paths))),
                    "path_length": int(layer.get("path_length", 0)),
                    "_approx_luminance": float(luminance),
                },
            )

            norm_layers.append(paint_layer)

        return norm_layers

    def derive_layer_execution(
        self,
        layer: PaintLayer,
        *,
        default_tool: str,
        default_pressure: float,
        z_down: float,
        z_up: Optional[float] = None,
        clean_interval: Optional[int] = None,
    ) -> PaintLayer:
        """Leitet ausführbare Parameter für eine Farbschicht ab.

        Die Analyse liefert umfangreiche Kennwerte (Stage, Technik, Stärken/Schwächen).
        Diese Funktion übersetzt sie in konkrete Maschinenparameter: Werkzeugwahl,
        Druck, Z-Höhen, Anzahl der Glaze-Pässe und Reinigungsintervalle.
        """

        metadata = layer.metadata

        stage = layer.stage or "mid"
        technique = layer.technique or ""
        shading = layer.shading or ""
        coverage = float(layer.coverage)
        detail_strength = float(metadata.get("detail_strength", 0.0))
        highlight_strength = float(metadata.get("highlight_strength", 0.0))
        shadow_strength = float(metadata.get("shadow_strength", 0.0))
        texture_strength = float(metadata.get("texture_strength", 0.0))
        contrast_strength = float(metadata.get("contrast_strength", 0.0))

        stage_to_tool = layer.tool
        tool = stage_to_tool if isinstance(stage_to_tool, str) else default_tool

        if shading == "highlight":
            tool = "highlight_brush"
            if not technique:
                technique = "luminous_glazing"
        elif shading == "shadow":
            tool = "shadow_brush"
            if not technique:
                technique = "shadow_glaze"

        brush_meta = layer.brush or self.get_brush_settings(tool)
        preset_pressure = brush_meta.get("default_pressure")
        if preset_pressure is None:
            preset_pressure = self._get_brush_pressure(tool)
        base_pressure = float(default_pressure)
        if preset_pressure is not None:
            try:
                base_pressure = 0.6 * base_pressure + 0.4 * float(preset_pressure)
            except (TypeError, ValueError):
                pass

        pressure = base_pressure
        if stage == "background":
            pressure *= 1.0 + 0.25 * np.clip(coverage - 0.25, -0.25, 0.5)
            pressure *= 1.0 + 0.1 * np.clip(shadow_strength - 0.4, 0.0, 0.6)
        elif stage == "detail":
            pressure *= 0.75
            pressure *= 1.0 - 0.2 * np.clip(detail_strength - 0.3, 0.0, 0.7)
        else:  # mid
            pressure *= 1.0 + 0.1 * np.clip(texture_strength - 0.4, 0.0, 0.6)

        if technique in {"luminous_glazing", "shadow_glaze"}:
            pressure *= 0.85
        if technique in {"precision_strokes", "cross_hatching"}:
            pressure *= 0.9
        if shading == "highlight":
            pressure *= 0.9
        elif shading == "shadow":
            pressure *= 1.05

        pressure = float(np.clip(pressure, 0.1, 1.0))

        layer_z_down = float(z_down)
        if stage == "background":
            layer_z_down += 0.25 * np.clip(coverage, 0.0, 0.6)
        elif stage == "detail":
            layer_z_down -= 0.4

        if technique == "luminous_glazing":
            layer_z_down -= 0.2
        elif technique == "shadow_glaze":
            layer_z_down += 0.2
        elif technique == "vibrant_impasto":
            layer_z_down += 0.15

        layer_z_down = max(0.1, layer_z_down)
        z_up_base = float(self.z_up_default if z_up is None else z_up)
        layer_z_up = float(z_up_base)
        if stage == "detail":
            layer_z_up = max(z_up_base, layer_z_down + 2.0)

        base_passes = max(1, int(self.num_glaze_passes))
        if technique in {"precision_strokes", "cross_hatching"}:
            passes = 1
        elif technique in {"luminous_glazing", "shadow_glaze"}:
            passes = max(base_passes, 3 if highlight_strength > 0.5 else 2)
        elif stage == "detail":
            passes = max(1, min(base_passes, 2))
        else:
            passes = base_passes

        base_clean = (
            int(clean_interval)
            if clean_interval is not None
            else int(self.clean_interval_default)
        )
        base_clean = max(1, base_clean)

        clean_interval_layer = base_clean
        if tool == "fine_brush" or stage == "detail":
            clean_interval_layer = min(clean_interval_layer, 2)
        elif coverage > 0.4:
            clean_interval_layer = max(clean_interval_layer, 6)
        elif technique == "vibrant_impasto":
            clean_interval_layer = max(clean_interval_layer, 5)

        needs_cleaning = bool(brush_meta.get("needs_cleaning", True))

        layer.brush = dict(brush_meta)
        layer.tool = tool
        layer.opacity = float(layer.brush.get("opacity", layer.opacity))
        layer.pressure = pressure
        layer.z_down = layer_z_down
        layer.z_up = layer_z_up
        layer.passes = passes
        layer.clean_interval = clean_interval_layer
        layer.needs_cleaning = needs_cleaning
        layer.technique = technique or layer.technique
        layer.stage = stage
        layer.shading = shading or layer.shading

        return layer





    def multi_layer_paintcode(
        self,
        normalized_layers: Iterable[PaintLayer],
        *,
        tool_name: str,
        pressure: float,
        z_up: float,
        z_down: float,
        clean_interval: Optional[int] = None
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

        if clean_interval is None:
            clean_interval = int(self.clean_interval_default)

        pal = self.machine_cfg.get("palette_station", {"x": 0, "y": 0, "z": z_down})
        wash = self.machine_cfg.get("wash_station", {"x": 0, "y": 0, "z": z_up})

        pal_x = float(pal.get("x", 0.0))
        pal_y = float(pal.get("y", 0.0))
        pal_z = float(pal.get("z", z_down))

        wash_x = float(wash.get("x", 0.0))
        wash_y = float(wash.get("y", 0.0))
        wash_z = float(wash.get("z", z_up))

        ordered_layers = sorted(normalized_layers, key=lambda layer: layer.sort_depth_key())

        prepared_layers: List[PaintLayer] = []
        for layer in ordered_layers:
            if not layer.mm_paths:
                continue

            prepared_layers.append(
                self.derive_layer_execution(
                    layer,
                    default_tool=tool_name,
                    default_pressure=pressure,
                    z_down=z_down,
                    z_up=z_up,
                    clean_interval=clean_interval,
                )
            )

        for layer_idx, layer in enumerate(prepared_layers):
            color_rgb = tuple(layer.color_rgb)
            mm_paths = layer.mm_paths

            layer_tool = layer.tool or tool_name
            layer_pressure = layer.pressure or pressure
            layer_z_up = layer.z_up or z_up
            layer_z_down = layer.z_down or z_down
            layer_passes = max(int(layer.passes or 1), 1)
            layer_clean_interval = int(layer.clean_interval or clean_interval)
            layer_needs_clean = bool(layer.needs_cleaning)
            layer_stage = layer.stage
            layer_technique = layer.technique
            layer_shading = layer.shading

            lines.append("; ===============================")
            lines.append(
                f"; Farb-Layer {layer_idx + 1}  RGB={color_rgb}"
            )
            if layer_stage:
                lines.append(f"; Stage: {layer_stage}")
            if layer_technique:
                lines.append(f"; Technik: {layer_technique}")
            if layer_shading:
                lines.append(f"; Shading: {layer_shading}")
            lines.append(f"; TOOL {layer_tool}")
            lines.append("; ===============================")

            for pass_idx in range(layer_passes):
                lines.append(
                    f"; --- Pass {pass_idx + 1}/{layer_passes} ({layer_tool}) ---"
                )
                lines.append(
                    f"GOTO_PALETTE X{pal_x:.2f} Y{pal_y:.2f} Z{layer_z_up:.2f}"
                )
                lines.append(
                    f"LOAD_COLOR R{color_rgb[0]} G{color_rgb[1]} B{color_rgb[2]}"
                )
                lines.append(
                    f"DIP_TOOL X{pal_x:.2f} Y{pal_y:.2f} Z{pal_z:.2f}"
                )
                lines.append(f"TOOL {layer_tool}")
                lines.append(f"PRESSURE {layer_pressure:.2f}")
                lines.append("")

                local_count_since_clean = 0

                for path in mm_paths:
                    if len(path) < 2:
                        continue

                    start_x, start_y = path[0]

                    lines.append(f"; Stroke {stroke_global_id}")
                    lines.append(
                        f"Z_UP X{start_x:.2f} Y{start_y:.2f} Z{layer_z_up:.2f}"
                    )
                    lines.append(
                        f"Z_DOWN X{start_x:.2f} Y{start_y:.2f} Z{layer_z_down:.2f}"
                    )
                    lines.append(
                        f"PEN_DOWN X{start_x:.2f} Y{start_y:.2f}"
                    )

                    for (x_mm, y_mm) in path[1:]:
                        lines.append(f"MOVE X{x_mm:.2f} Y{y_mm:.2f}")

                    lines.append("PEN_UP")

                    end_x, end_y = path[-1]
                    lines.append(
                        f"Z_UP X{end_x:.2f} Y{end_y:.2f} Z{layer_z_up:.2f}"
                    )
                    lines.append("")

                    stroke_global_id += 1
                    local_count_since_clean += 1

                    if (
                        layer_needs_clean
                        and layer_clean_interval > 0
                        and local_count_since_clean >= layer_clean_interval
                    ):
                        lines.append(
                            f"GOTO_WASH X{wash_x:.2f} Y{wash_y:.2f} Z{layer_z_up:.2f}"
                        )
                        lines.append(
                            f"CLEAN_TOOL X{wash_x:.2f} Y{wash_y:.2f} Z{wash_z:.2f}"
                        )
                        lines.append("")
                        local_count_since_clean = 0

                if layer_needs_clean:
                    lines.append(
                        f"GOTO_WASH X{wash_x:.2f} Y{wash_y:.2f} Z{layer_z_up:.2f}"
                    )
                    lines.append(
                        f"CLEAN_TOOL X{wash_x:.2f} Y{wash_y:.2f} Z{wash_z:.2f}"
                    )
                    lines.append("")

        if not lines:
            return "; (keine Pfade gefunden)"

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
