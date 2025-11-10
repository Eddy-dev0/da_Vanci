import logging
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QLabel,
    QToolBar,
    QFileDialog,
    QMessageBox,
    # plus: in den Builder-Funktionen importieren wir dynamisch weitere Widgets,
    # das ist ok.
)
from PySide6.QtGui import QIcon, QAction, QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple
from painterslicer.image_analysis.analyzer import ImageAnalyzer
from painterslicer.image_analysis.pipeline import PaintingPipeline, PipelineResult
from painterslicer.slicer.slicer_core import PainterSlicer
from painterslicer.slicer.layers import PaintLayer
from painterslicer.utils.brush_tool import BrushTool


LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Basis-Fenster-Setup ---
        self.setWindowTitle("PainterSlicer 🎨🤖")
        self.resize(1200, 800)

        # Analyzer / Slicer Objekte
        self.analyzer = ImageAnalyzer()
        self.slicer = PainterSlicer()
        self.pipeline = PaintingPipeline()

        self.paint_styles: Dict[str, Dict[str, Any]] = self._build_style_profiles()
        self.selected_style_key: str = "Original"
        if not self.paint_styles:
            self.paint_styles = {
                "Standard": {"analyzer": {}, "slicer": {}, "description": ""}
            }
        if self.selected_style_key not in self.paint_styles:
            self.selected_style_key = next(iter(self.paint_styles.keys()))
        self.active_style_profile: Dict[str, Any] = {}
        self.active_pipeline_profile: Dict[str, Any] = {}
        self._active_brush_overrides: Dict[str, Dict[str, Any]] = {}
        self._brush_tool_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], BrushTool] = {}
        self._apply_style_profile(self.selected_style_key)

        self.last_pipeline_result: Optional[PipelineResult] = None
        self.last_pipeline_summary: List[str] = []
        self.pipeline_stroke_plan_mm: List[Dict[str, Any]] = []
        self._superres_forced_off_reason: Optional[str] = None

        # Animation zurücksetzen
        # --- Animation / Preview State ---
        from PySide6.QtCore import QTimer, Qt

#        self.speed_slider.valueChanged.connect(self._update_animation_speed)

        # ...
        self.anim_in_progress = False
        self.anim_stroke_index = 0
        self.anim_point_index = 0  # nächstes Segment innerhalb eines Strokes
        self.current_highlight_segment: Optional[Tuple[int, int]] = None
        self.paint_strokes_timeline = []
        self.preview_canvas_pixmap = None
        self._current_image_size: Optional[Tuple[int, int]] = None
        self._last_preview_display: Optional[QPixmap] = None
        self.last_planning_result = None

        # QTimer EINMAL erstellen und dauerhaft behalten
        self.anim_timer = QTimer(self)
        self.anim_timer.setTimerType(Qt.PreciseTimer)  # präziser Takt
        self.anim_timer.setSingleShot(False)  # WICHTIG: NICHT SingleShot
        self.anim_timer.setInterval(30)  # Default, wird bei Play gesetzt
        self.anim_timer.timeout.connect(self.animation_step)
        self.anim_timer.stop()

        # ---- User-Einstellungen für Mal-Parameter (MÜSSEN früh kommen!) ----
        # Standardwerte, können später im Machine-Tab geändert werden
        self.selected_tool = "fine_brush"
        self.paint_pressure = 0.30       # 0..1
        self.z_down = 5.0                # mm (Arbeits- / Malhöhe)
        self.z_up = 50.0                 # mm (Sichere Verfahrhöhe)


        # Pfade für Vorschau (in mm normalisiert)
        # Liste von Strokes, jeder Stroke ist Liste [(x_mm, y_mm), ...]
        self.last_mm_paths = []



        # Zentraler Widget-Container
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        central.setLayout(layout)

        # Tabs oben
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tabs aufbauen
        self.import_tab = self._build_import_tab()
        self.analysis_tab = self._build_analysis_tab()
        self.slice_tab = self._build_slice_tab()
        self.preview_tab = self._build_preview_tab()
        self.machine_tab = self._build_machine_tab()

        self.tabs.addTab(self.import_tab, "Import")
        self.tabs.addTab(self.analysis_tab, "Analyse")
        self.tabs.addTab(self.slice_tab, "Slice")
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.machine_tab, "Machine")

        # Toolbar aufbauen
        self._build_toolbar()

        # aktuell geladener Bildpfad + letzter Export
        self.current_image_path = None
        self.last_paintcode_export = None



    # ---------- UI-Bausteine ----------

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Bild öffnen
        open_image_action = QAction(QIcon(), "Bild öffnen", self)
        open_image_action.triggered.connect(self.action_load_image)
        toolbar.addAction(open_image_action)

        # Analysieren
        analyze_action = QAction(QIcon(), "Analysieren", self)
        analyze_action.triggered.connect(self.action_run_analysis)
        toolbar.addAction(analyze_action)

        # Slice planen
        slice_action = QAction(QIcon(), "Slice planen", self)
        slice_action.triggered.connect(self.action_slice_plan)
        toolbar.addAction(slice_action)

        # PaintCode exportieren
        export_action = QAction(QIcon(), "PaintCode exportieren", self)
        export_action.triggered.connect(self.action_export_paintcode)
        toolbar.addAction(export_action)

        # Mit Maschine verbinden (Stub)
        connect_action = QAction(QIcon(), "Mit Maschine verbinden", self)
        connect_action.triggered.connect(self.action_connect_machine)
        toolbar.addAction(connect_action)




    def _build_import_tab(self):
        """Tab 0: Import – Bild laden und anzeigen."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Bildanzeige-Label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444;")
        self.image_label.setMinimumSize(400, 300)  # damit man was sieht
        layout.addWidget(self.image_label)

        # Info-Text (Pfad / Hinweise)
        self.import_info_label = QLabel("Noch kein Bild geladen.\nBenutze oben 'Bild öffnen'.")
        self.import_info_label.setAlignment(Qt.AlignCenter)
        self.import_info_label.setStyleSheet("font-size: 16px; color: #aaa; padding: 20px;")
        layout.addWidget(self.import_info_label)

        return tab



    def _build_slice_tab(self):
        """
        Tab: Slice – zeigt den geplanten Mal-Ablauf (Schritte & Tools).
        """
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # WICHTIG: alles klein geschrieben: self.slice_label
        self.slice_label = QLabel("Noch kein Slice-Plan generiert.")
        self.slice_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.slice_label.setStyleSheet(
            "background-color: #1e1e1e; color: #ccc; border: 1px solid #444; "
            "font-family: Consolas, monospace; font-size: 14px; padding: 12px;"
        )
        self.slice_label.setMinimumSize(400, 300)
        self.slice_label.setWordWrap(True)

        layout.addWidget(self.slice_label)
        return tab



    def _build_analysis_tab(self):
        """
        Tab 1: Analyse – zeigt z. B. die Kantenmaske vom Analyzer.
        """
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Hier zeigen wir die verarbeitete Version (z. B. Kanten)
        self.analysis_label = QLabel("Noch keine Analyse durchgeführt.")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.analysis_label.setStyleSheet("background-color: #1e1e1e; color: #aaa; border: 1px solid #444;")
        self.analysis_label.setMinimumSize(400, 300)

        layout.addWidget(self.analysis_label)

        return tab




    def _build_preview_tab(self):
        """
        Tab: Preview – farbige Anzeige + Animation + Steuerung.
        """
        from PySide6.QtWidgets import (
            QVBoxLayout,
            QHBoxLayout,
            QPushButton,
            QSlider,
            QLabel,
        )
        from PySide6.QtCore import Qt

        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Zeichenfläche
        self.preview_label = QLabel("Noch keine Vorschau.\nBitte zuerst 'Slice planen'.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #000; color: #aaa; border: 1px solid #444;"
        )
        self.preview_label.setMinimumSize(400, 400)
        layout.addWidget(self.preview_label)

        # Playback Controls
        controls_row = QHBoxLayout()

        self.btn_play = QPushButton("Play")
        self.btn_show_all = QPushButton("Fertig anzeigen")

        controls_row.addWidget(self.btn_play)
        controls_row.addWidget(self.btn_show_all)

        layout.addLayout(controls_row)

        # Slider für Fortschritt
        slider_row = QHBoxLayout()
        self.progress_label = QLabel("0 / 0")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(0)  # setzen wir später nach slice
        self.progress_slider.setValue(0)

        slider_row.addWidget(QLabel("Fortschritt:"))
        slider_row.addWidget(self.progress_slider)
        slider_row.addWidget(self.progress_label)

        layout.addLayout(slider_row)

        # Geschwindigkeit (Timer-Intervall)
        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)    # schnell
        self.speed_slider.setMaximum(200)  # langsam
        self.speed_slider.setValue(30)     # default
        speed_row.addWidget(QLabel("Geschwindigkeit"))
        speed_row.addWidget(self.speed_slider)

        layout.addLayout(speed_row)

        # Verhalten der Buttons / Slider
        self.btn_play.clicked.connect(self.start_preview_animation)
        self.btn_show_all.clicked.connect(self.render_preview_full_colored)
        self.progress_slider.valueChanged.connect(self.scrub_preview_to)

        layout.addStretch(1)

        self.speed_slider.valueChanged.connect(self._update_animation_speed)

        return tab









    def _placeholder_tab(self, text: str):
        """Platzhalter für Tabs, die wir noch bauen werden."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(label)

        return tab

    # ---------- Aktionen (Toolbar Funktionen) ----------

    def action_load_image(self):
        """
        Bild vom PC auswählen, anzeigen und Pfad merken.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild wählen",
            os.path.expanduser("~"),
            "Bilder (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not file_path:
            return  # User hat Abbrechen gedrückt

        # Pfad speichern
        self.current_image_path = file_path

        # Bild ins Label laden
        pixmap = QPixmap(file_path)

        # Originalgröße merken, damit Preview/Exports das Ausgangsformat behalten
        if not pixmap.isNull():
            self._current_image_size = (pixmap.width(), pixmap.height())
        else:
            self._current_image_size = None

        if pixmap.isNull():
            # Falls das Bild nicht geladen werden kann
            self.image_label.setText("Fehler beim Laden des Bildes.")
            self.import_info_label.setText("Ungültiges Bildformat?")
            return

        # Bild runterskalieren, damit es in den verfügbaren Bereich passt
        # Wir nehmen die aktuelle Größe vom Label als Referenz.
        target_w = max(self.image_label.width(), 400)
        target_h = max(self.image_label.height(), 300)

        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Ins UI einsetzen
        self.image_label.setPixmap(scaled)
        self.import_info_label.setText(f"Geladenes Bild:\n{file_path}")





    def action_slice_plan(self):
        """
        Farblayer erzeugen, normalisieren, PaintCode generieren,
        Preview-Daten (Timeline) vorbereiten.
        """

        if not self.current_image_path:
            QMessageBox.warning(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return

        # 1. Farb- und Layer-Planung (liefert Masken + Farblayer)
        analyzer_style: Dict[str, Any] = {}
        slicer_style: Dict[str, Any] = {}
        pipeline_style: Dict[str, Any] = {}
        if getattr(self, "active_style_profile", None):
            analyzer_style = dict(self.active_style_profile.get("analyzer", {}))
            slicer_style = dict(self.active_style_profile.get("slicer", {}))
            self.slicer.apply_style_profile(slicer_style)
            pipeline_style = dict(self.active_style_profile.get("pipeline", {}))

        planning_source: Any = self.current_image_path
        pipeline_summary: List[str] = []
        if pipeline_style:
            try:
                planning_source, pipeline_summary = self._prepare_planning_source(
                    self.current_image_path,
                    pipeline_style,
                )
            except Exception as exc:
                planning_source = self.current_image_path
                self.last_pipeline_result = None
                self.last_pipeline_summary = []
                self.pipeline_stroke_plan_mm = []
                pipeline_summary = [
                    "High-Fidelity-Pipeline konnte nicht vollständig ausgeführt werden.",
                    f"Fehler: {exc}",
                ]

        try:
            planning_kwargs: Dict[str, Any] = {
                "k_min": analyzer_style.get("k_min", 12),
                "k_max": analyzer_style.get("k_max", 28),
                "style_profile": analyzer_style,
            }
            if analyzer_style.get("k_colors") is not None:
                planning_kwargs["k_colors"] = analyzer_style.get("k_colors")

            planning_result = self.analyzer.plan_painting_layers(
                planning_source,
                **planning_kwargs,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Analyse fehlgeschlagen",
                f"Planung konnte nicht erzeugt werden:\n{exc}",
            )
            return

        self.last_planning_result = planning_result
        layer_masks = planning_result.get("layer_masks", {}) or {}
        color_layers = planning_result.get("layers", []) or []

        paint_plan: dict = {"steps": []}
        try:
            plan_result = self.slicer.generate_paint_plan(layer_masks)
            if isinstance(plan_result, dict):
                paint_plan = plan_result
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Planung unvollständig",
                f"Der Slice-Plan konnte nicht vollständig erstellt werden:\n{exc}",
            )

        if not color_layers:
            QMessageBox.warning(
                self,
                "Keine Layer",
                "Die Analyse hat keine Farbschichten ergeben.",
            )
            return

        plan_lines = ["Geplanter Ablauf (Logik):\n"]
        step_id = 1
        for step in paint_plan.get("steps", []):
            plan_lines.append(
                f"Schritt {step_id}: {step['layer']}\n"
                f"  Werkzeug: {step['tool']}\n"
                f"  Aktiv?   {'ja' if step['mask_present'] else 'nein'}\n"
                f"  Zweck:    {step['comment']}\n"
            )
            step_id += 1

        # 2. In mm skalieren
        normalized_layers: List[PaintLayer] = self.slicer.normalize_color_layers_to_mm(
            self.current_image_path,
            color_layers,
            style_key=self.selected_style_key,
        )

        normalized_layers.sort(key=lambda layer: layer.sort_depth_key())

        clean_interval = slicer_style.get("clean_interval") if slicer_style else None

        execution_profiles: List[PaintLayer] = []
        for layer in normalized_layers:
            execution_profiles.append(
                self.slicer.derive_layer_execution(
                    layer,
                    default_tool=self.selected_tool,
                    default_pressure=self.paint_pressure,
                    z_down=self.z_down,
                    z_up=self.z_up,
                    clean_interval=clean_interval,
                )
            )
        # normalized_layers = [
        #   { "color_rgb": (r,g,b), "mm_paths": [ [ (x_mm,y_mm)... ], ... ] },
        #   ...
        # ]

        # 4. Multi-Layer PaintCode erzeugen
        tool_name = self.selected_tool
        pressure = self.paint_pressure
        z_up = self.z_up
        z_down = self.z_down

        paintcode_multi = self.slicer.multi_layer_paintcode(
            execution_profiles,
            tool_name=tool_name,
            pressure=pressure,
            z_up=z_up,
            z_down=z_down,
            clean_interval=clean_interval
        )

        # 5. Timeline für Preview aufbauen
        # Eine flache Liste aller Strokes in Mal-Reihenfolge,
        # jeweils mit Farbe und Punkten.
        self.paint_strokes_timeline = []
        for profile in execution_profiles:
            mm_paths = profile.mm_paths
            for pass_idx in range(profile.passes):
                for stroke in mm_paths:
                    if len(stroke) < 2:
                        continue
                    self.paint_strokes_timeline.append(
                        {
                            "color_rgb": profile.color_rgb,
                            "points": stroke,
                            "stage": profile.stage,
                            "technique": profile.technique,
                            "label": profile.label,
                            "shading": profile.shading,
                            "tool": profile.tool,
                            "pressure": profile.pressure,
                            "pass_index": pass_idx,
                            "passes": profile.passes,
                            "z_down": profile.z_down,
                            "z_up": profile.z_up,
                            "brush": profile.brush,
                            "opacity": profile.opacity,
                        }
                    )

        # 6. Für "alte" Preview-APIs (render_preview_frame etc.) behalten wir
        # noch eine einfache Liste aller Pfade ohne Farbe
        self.last_mm_paths = [s["points"] for s in self.paint_strokes_timeline]

        # 7. Für Export merken
        self.last_paintcode_export = paintcode_multi

        # 8. Slice-Tab Text zusammensetzen
        lines_out = []
        if self.selected_style_key:
            style_profile = self.paint_styles.get(self.selected_style_key, {})
            lines_out.append(f"Aktiver Malstil: {self.selected_style_key}")
            desc = style_profile.get("description")
            if desc:
                lines_out.append(desc)
            lines_out.append("")
        if pipeline_summary:
            lines_out.append("--- High-Fidelity Pipeline ---")
            lines_out.extend(pipeline_summary)
            lines_out.append("")
        lines_out.extend(plan_lines)

        if execution_profiles:
            lines_out.append("")
            lines_out.append("--- Layer Execution Mapping ---")
            for idx, profile in enumerate(execution_profiles, start=1):
                color_rgb = profile.color_rgb
                lines_out.append(
                    f"Layer {idx} (Stage {profile.stage}, Technik {profile.technique}):"
                )
                lines_out.append(f"  Farbe: RGB{color_rgb}")
                lines_out.append(
                    f"  Werkzeug: {profile.tool}  Druck: {profile.pressure:.2f}  Pässe: {profile.passes}"
                )
                lines_out.append(
                    f"  Z_down: {profile.z_down:.2f}  Z_up: {profile.z_up:.2f}"
                )
                lines_out.append(
                    f"  Coverage: {profile.coverage:.3f}  Highlights: {profile.metadata.get('highlight_strength', 0.0):.2f}  Schatten: {profile.metadata.get('shadow_strength', 0.0):.2f}"
                )
                lines_out.append("")
        lines_out.append("\n--- Farb-Layer Info ---\n")
        lines_out.append(f"Anzahl Farblayer: {len(normalized_layers)}\n")

        for idx, layer in enumerate(normalized_layers):
            rgb = layer.color_rgb
            coverage_pct = float(layer.coverage * 100.0)
            stage = layer.stage or "?"
            order = layer.order if layer.order is not None else idx
            tool = layer.tool or "-"
            technique = layer.technique or "-"
            path_count = len(layer.mm_paths)
            lines_out.append(
                f"Layer {idx+1:02d} (Order {order}, Stage {stage}): RGB={rgb}, "
                f"Coverage={coverage_pct:.1f}%, Pfade={path_count}, Werkzeug={tool}, "
                f"Technik={technique}"
            )
            lines_out.append(
                "    detail={detail:.2f} mid={mid:.2f} background={bg:.2f} "
                "texture={tex:.2f} highlight={hi:.2f} shadow={sh:.2f} "
                "contrast={co:.2f} colorVar={cv:.2f}".format(
                    detail=float(layer.metadata.get("detail_ratio", 0.0)),
                    mid=float(layer.metadata.get("mid_ratio", 0.0)),
                    bg=float(layer.metadata.get("background_ratio", 0.0)),
                    tex=float(layer.metadata.get("texture_strength", 0.0)),
                    hi=float(layer.metadata.get("highlight_strength", 0.0)),
                    sh=float(layer.metadata.get("shadow_strength", 0.0)),
                    co=float(layer.metadata.get("contrast_strength", 0.0)),
                    cv=float(layer.metadata.get("color_variance_strength", 0.0)),
                )
            )


        lines_out.append("\n--- PaintCode (Multi-Layer) ---\n")
        lines_out.append(paintcode_multi)

        final_text = "\n".join(lines_out)
        self.slice_label.setText(final_text)

        # 9. Animation State zurücksetzen
        # Animation Startzustand
        # Animations-Zustand initialisieren
        # Animation-Reset
        self.anim_stroke_index = 0
        self.anim_point_index = 0
        self.anim_in_progress = False
        self.current_highlight_segment = None

        # Fortschritt-Slider setzen (0..len-1)
        if hasattr(self, "progress_slider"):
            self.progress_slider.setMinimum(0)
            self.progress_slider.setMaximum(max(len(self.paint_strokes_timeline) - 1, 0))
            self.progress_slider.setValue(0)

        # erstes Bild: noch nix gemalt
        pm0 = self._render_full_state_at(-1)  # liefert schwarze Fläche
        if pm0 is not None:
            self.preview_canvas_pixmap = pm0
            self._display_preview_pixmap(pm0)
        self._update_progress_ui()






    def action_run_analysis(self):
        """
        Nimmt das aktuell geladene Bild,
        lässt den Analyzer eine Kantenmaske bauen
        und zeigt diese Maske im Analyse-Tab an.
        """
        if not self.current_image_path:
            QMessageBox.warning(self, "Kein Bild", "Bitte zuerst ein Bild laden.")
            return

        if not hasattr(self.analyzer, "analyze_for_preview"):
            QMessageBox.critical(
                self,
                "Analyse nicht verfügbar",
                "Der Bild-Analyzer unterstützt keine Vorschau-Analyse.",
            )
            return

        # 1) Analyse laufen lassen
        edge_mask = self.analyzer.analyze_for_preview(self.current_image_path)

        if edge_mask is None:
            QMessageBox.warning(self, "Analyse fehlgeschlagen", "Konnte keine Kantenmaske erzeugen.")
            return

        # edge_mask ist ein 2D numpy array (uint8), Werte 0..255.
        h, w = edge_mask.shape

        # QImage übernimmt bei dieser Konstruktion keinen Besitz an den Rohdaten.
        # Wenn edge_mask am Ende dieser Funktion aus dem Scope fällt, zeigt QImage
        # auf Speicher, der bereits freigegeben wurde -> Crash (0xCFFFFFFF).
        # Wir halten deshalb eine Kopie als Instanzattribut fest, so dass die
        # Daten während der UI-Lebensdauer gültig bleiben.
        self._analysis_edge_mask = edge_mask.copy(order="C")

        # 2) In ein QImage konvertieren
        bytes_per_line = int(self._analysis_edge_mask.strides[0])

        qimg_view = QImage(
            self._analysis_edge_mask.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_Grayscale8
        )

        # QImage kopiert die Daten nicht automatisch. Durch copy() erhalten wir
        # ein eigenständiges QImage, dessen Speicher von Qt verwaltet wird. Damit
        # vermeiden wir Zugriffe auf bereits freigegebenen NumPy-Speicher, die
        # zuvor zu Abstürzen (0xCFFFFFFF) beim Tab-Wechsel führten.
        qimg = qimg_view.copy()
        self._analysis_qimage = qimg  # Referenz halten, falls Qt lazy shared.

        pixmap = QPixmap.fromImage(qimg)

        # 3) Skalieren, damit es schön in unser Label passt
        target_w = max(self.analysis_label.width(), 400)
        target_h = max(self.analysis_label.height(), 300)

        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # 4) ins UI einsetzen
        self.analysis_label.setPixmap(scaled)
        self.analysis_label.setText("")  # alten Text löschen

        # 5) direkt zum Analyse-Tab springen
        index = self.tabs.indexOf(self.analysis_tab)
        if index != -1:
            self.tabs.setCurrentIndex(index)



    def action_export_paintcode(self):
        """
        Speichert den zuletzt generierten PaintCode als Datei.
        Format: .paintcode (reiner Text)
        """

        # Haben wir überhaupt schon gesliced?
        if not self.last_paintcode_export:
            QMessageBox.warning(
                self,
                "Nichts zu exportieren",
                "Bitte zuerst 'Slice planen' ausführen, um PaintCode zu erzeugen."
            )
            return

        # Dateidialog öffnen
        default_name = "job.paintcode"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "PaintCode speichern",
            default_name,
            "PaintCode Dateien (*.paintcode);;Textdateien (*.txt);;Alle Dateien (*)"
        )

        # Falls der User abbricht
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.last_paintcode_export)

            QMessageBox.information(
                self,
                "Export erfolgreich",
                f"PaintCode gespeichert unter:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler beim Speichern",
                f"Konnte Datei nicht speichern:\n{e}"
            )


    def action_connect_machine(self):
        """
        Später:
        - versucht seriell Verbindung zur Mal-Maschine herzustellen
        - checkt Firmware
        Für jetzt: nur Dummy.
        """
        QMessageBox.information(
            self,
            "Maschine",
            "Maschinen-Verbindung (serial/USB) kommt später hier rein."
        )
    def resizeEvent(self, event):
        """
        Wenn das Fenster größer/kleiner gezogen wird,
        skalieren wir die Bildvorschau neu.
        """
        super().resizeEvent(event)

        if not hasattr(self, "current_image_path"):
            return

        if not self.current_image_path:
            return

        pixmap = QPixmap(self.current_image_path)
        if pixmap.isNull():
            return

        target_w = max(self.image_label.width(), 400)
        target_h = max(self.image_label.height(), 300)

        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

        if self._last_preview_display is not None:
            self._display_preview_pixmap(self._last_preview_display)

    def _build_machine_tab(self):
        """
        Tab: Machine – UI für Werkzeug/Druck/Z-Parameter.
        Diese Werte steuern später den PaintCode.
        """
        from PySide6.QtWidgets import (
            QVBoxLayout,
            QFormLayout,
            QComboBox,
            QDoubleSpinBox,
            QPushButton,
            QGroupBox,
        )

        tab = QWidget()
        outer_layout = QVBoxLayout()
        tab.setLayout(outer_layout)

        # Gruppe mit den Einstellfeldern
        group_params = QGroupBox("Mal-Parameter / Werkzeug")
        form_layout = QFormLayout()
        group_params.setLayout(form_layout)

        # Werkzeug-Auswahl
        self.style_combo = QComboBox()
        self.style_combo.addItems(list(self.paint_styles.keys()))
        self.style_combo.setCurrentText(self.selected_style_key)
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        form_layout.addRow("Malstil:", self.style_combo)

        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["broad_brush", "medium_brush", "fine_brush", "sponge"])
        self.tool_combo.setCurrentText(self.selected_tool)
        form_layout.addRow("Werkzeug:", self.tool_combo)

        # Druck
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(0.0, 1.0)
        self.pressure_spin.setSingleStep(0.05)
        self.pressure_spin.setValue(self.paint_pressure)
        form_layout.addRow("Druck (0..1):", self.pressure_spin)

        # Z unten (Maltiefe)
        self.zdown_spin = QDoubleSpinBox()
        self.zdown_spin.setRange(0.0, 200.0)
        self.zdown_spin.setSingleStep(0.5)
        self.zdown_spin.setValue(self.z_down)
        form_layout.addRow("Z unten (mm):", self.zdown_spin)

        # Z oben (Hubhöhe)
        self.zup_spin = QDoubleSpinBox()
        self.zup_spin.setRange(0.0, 200.0)
        self.zup_spin.setSingleStep(0.5)
        self.zup_spin.setValue(self.z_up)
        form_layout.addRow("Z oben (mm):", self.zup_spin)

        # Button zum Übernehmen
        apply_btn = QPushButton("Parameter übernehmen")
        form_layout.addRow("", apply_btn)

        def apply_settings():
            # Werte aus den Widgets zurück in den MainWindow-State schreiben
            self.selected_tool = self.tool_combo.currentText()
            self.paint_pressure = float(self.pressure_spin.value())
            self.z_down = float(self.zdown_spin.value())
            self.z_up = float(self.zup_spin.value())

        apply_btn.clicked.connect(apply_settings)

        outer_layout.addWidget(group_params)

        self.style_description_label = QLabel()
        self.style_description_label.setWordWrap(True)
        self.style_description_label.setStyleSheet("color: #ccc; font-size: 12px; padding: 4px 0;")
        outer_layout.addWidget(self.style_description_label)
        self._update_style_description()

        info_label = QLabel(
            "Diese Werte fließen in den PaintCode ein.\n"
            "Als Nächstes: COM-Port Verbindung und Farbstation/Waschlogik."
        )
        info_label.setStyleSheet("color: #aaa; font-size: 12px;")
        outer_layout.addWidget(info_label)

        outer_layout.addStretch(1)

        return tab



    def _build_style_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Definiert Malstil-Voreinstellungen für Analyse- und Slice-Pipeline."""

        return {
            "Studio - Schnell": {
                "description": (
                    "Ausgewogener Stil für schnelle Ergebnisse mit klaren Formen "
                    "und sichtbaren Konturen."
                ),
                "analyzer": {
                    "k_min": 14,
                    "k_max": 26,
                    "use_dither": True,
                    "min_path_length": 2,
                    "min_area_ratio": 0.00045,
                    "stroke_spacing_scale": 0.9,
                    "preserve_edge_strokes": True,
                    "detail_edge_boost": 1.1,
                    "edge_sensitivity": 1.0,
                    "background_stage_gain": 0.95,
                    "mid_stage_gain": 1.0,
                    "detail_stage_gain": 1.1,
                    "microtransition_boost": 1.0,
                    "super_sample_scale": 1.4,
                    "max_analysis_dimension": 1600,
                    "chroma_boost": 1.0,
                    "highlight_boost": 0.05,
                    "highlight_bias": 1.0,
                    "shadow_bias": 1.0,
                    "color_variance_bias": 1.0,
                },
                "slicer": {
                    "grid_mm": 0.26,
                    "num_glaze_passes": 3,
                    "clean_interval": 5,
                },
                "brushes": {
                    "broad_brush": {
                        "width_px": 42,
                        "opacity": 0.78,
                        "edge_softness": 0.4,
                        "flow": 0.65,
                    },
                    "medium_brush": {
                        "width_px": 24,
                        "opacity": 0.82,
                        "edge_softness": 0.35,
                        "flow": 0.78,
                    },
                    "fine_brush": {
                        "width_px": 12,
                        "opacity": 0.9,
                        "edge_softness": 0.25,
                        "flow": 0.9,
                    },
                    "highlight_brush": {
                        "width_px": 9,
                        "opacity": 0.97,
                        "edge_softness": 0.2,
                        "flow": 0.95,
                    },
                    "shadow_brush": {
                        "width_px": 10,
                        "opacity": 0.93,
                        "edge_softness": 0.26,
                        "flow": 0.9,
                    },
                },
            },
            "Galerie - Realismus": {
                "description": (
                    "Fein abgestimmter Realismus mit erhöhter Farbvielfalt, "
                    "weichen Übergängen und klar ausgearbeiteten Details."
                ),
                "analyzer": {
                    "k_min": 20,
                    "k_max": 36,
                    "use_dither": True,
                    "min_path_length": 2,
                    "min_area_ratio": 0.00032,
                    "stroke_spacing_scale": 0.7,
                    "preserve_edge_strokes": True,
                    "detail_edge_boost": 1.35,
                    "edge_sensitivity": 1.25,
                    "background_stage_gain": 0.9,
                    "mid_stage_gain": 1.05,
                    "detail_stage_gain": 1.35,
                    "microtransition_boost": 1.15,
                    "super_sample_scale": 1.7,
                    "max_analysis_dimension": 2048,
                    "chroma_boost": 1.08,
                    "highlight_boost": 0.12,
                    "highlight_bias": 1.15,
                    "shadow_bias": 1.05,
                    "color_variance_bias": 1.1,
                },
                "slicer": {
                    "grid_mm": 0.22,
                    "num_glaze_passes": 4,
                    "clean_interval": 4,
                },
                "brushes": {
                    "broad_brush": {
                        "width_px": 50,
                        "opacity": 0.82,
                        "edge_softness": 0.6,
                        "flow": 0.6,
                    },
                    "medium_brush": {
                        "width_px": 26,
                        "opacity": 0.86,
                        "edge_softness": 0.55,
                        "flow": 0.72,
                    },
                    "fine_brush": {
                        "width_px": 13,
                        "opacity": 0.92,
                        "edge_softness": 0.4,
                        "flow": 0.88,
                    },
                    "highlight_brush": {
                        "width_px": 9,
                        "opacity": 0.98,
                        "edge_softness": 0.24,
                        "flow": 0.95,
                    },
                    "shadow_brush": {
                        "width_px": 11,
                        "opacity": 0.92,
                        "edge_softness": 0.32,
                        "flow": 0.9,
                    },
                },
            },
            "Classic Style": {
                "description": (
                    "Zeitloser Klassik-Stil mit feinsten Übergängen, dichter "
                    "Schichtstruktur und präzise nachgezeichneten Kanten für "
                    "eine elegante Galerie-Anmutung."
                ),
                "analyzer": {
                    "k_min": 28,
                    "k_max": 56,
                    "use_dither": True,
                    "min_path_length": 2,
                    "min_area_ratio": 0.00018,
                    "stroke_spacing_scale": 0.45,
                    "preserve_edge_strokes": True,
                    "detail_edge_boost": 1.8,
                    "edge_sensitivity": 1.7,
                    "background_stage_gain": 0.82,
                    "mid_stage_gain": 1.15,
                    "detail_stage_gain": 1.7,
                    "microtransition_boost": 1.28,
                    "super_sample_scale": 1.9,
                    "max_analysis_dimension": 2300,
                    "chroma_boost": 1.14,
                    "highlight_boost": 0.18,
                    "highlight_bias": 1.35,
                    "shadow_bias": 1.15,
                    "color_variance_bias": 1.25,
                },
                "slicer": {
                    "grid_mm": 0.15,
                    "num_glaze_passes": 7,
                    "clean_interval": 2,
                },
                "brushes": {
                    "broad_brush": {
                        "width_px": 52,
                        "opacity": 0.84,
                        "edge_softness": 0.72,
                        "flow": 0.58,
                    },
                    "medium_brush": {
                        "width_px": 28,
                        "opacity": 0.9,
                        "edge_softness": 0.65,
                        "flow": 0.76,
                    },
                    "fine_brush": {
                        "width_px": 14,
                        "opacity": 0.94,
                        "edge_softness": 0.5,
                        "flow": 0.9,
                    },
                    "highlight_brush": {
                        "width_px": 9,
                        "opacity": 0.99,
                        "edge_softness": 0.22,
                        "flow": 0.96,
                    },
                    "shadow_brush": {
                        "width_px": 11,
                        "opacity": 0.94,
                        "edge_softness": 0.3,
                        "flow": 0.92,
                    },
                },
            },
            "Original": {
                "description": (
                    "Flaggschiff-Qualität mit maximaler Farbdifferenzierung, "
                    "ultrafeiner Detailabstufung, verstärkten Highlights und "
                    "erweitertem Layering für sichtbaren Qualitätszuwachs "
                    "gegenüber allen anderen Profilen."
                ),
                "analyzer": {
                    "k_min": 34,
                    "k_max": 72,
                    "use_dither": True,
                    "min_path_length": 2,
                    "min_area_ratio": 0.00012,
                    "stroke_spacing_scale": 0.38,
                    "preserve_edge_strokes": True,
                    "detail_edge_boost": 2.05,
                    "edge_sensitivity": 1.9,
                    "background_stage_gain": 0.78,
                    "mid_stage_gain": 1.18,
                    "detail_stage_gain": 1.95,
                    "microtransition_boost": 1.42,
                    "super_sample_scale": 2.1,
                    "max_analysis_dimension": 2560,
                    "chroma_boost": 1.22,
                    "highlight_boost": 0.24,
                    "highlight_bias": 1.55,
                    "shadow_bias": 1.2,
                    "color_variance_bias": 1.4,
                },
                "slicer": {
                    "grid_mm": 0.12,
                    "num_glaze_passes": 9,
                    "clean_interval": 2,
                },
                "brushes": {
                    "broad_brush": {
                        "width_px": 54,
                        "opacity": 0.85,
                        "edge_softness": 0.78,
                        "flow": 0.6,
                    },
                    "medium_brush": {
                        "width_px": 30,
                        "opacity": 0.9,
                        "edge_softness": 0.68,
                        "flow": 0.78,
                    },
                    "fine_brush": {
                        "width_px": 15,
                        "opacity": 0.96,
                        "edge_softness": 0.48,
                        "flow": 0.92,
                    },
                    "sponge": {
                        "width_px": 68,
                        "opacity": 0.58,
                        "edge_softness": 0.95,
                        "flow": 0.38,
                    },
                    "highlight_brush": {
                        "width_px": 10,
                        "opacity": 0.99,
                        "edge_softness": 0.2,
                        "flow": 0.97,
                    },
                    "shadow_brush": {
                        "width_px": 12,
                        "opacity": 0.95,
                        "edge_softness": 0.28,
                        "flow": 0.93,
                    },
                },
                "pipeline": {
                    "run_pipeline": True,
                    "enable_superres": True,
                    "superres_scale": 2,
                    "apply_guided_filter": True,
                    "guided_radius": 9,
                    "guided_eps": 1e-4,
                    "bilateral_diameter": 11,
                    "bilateral_sigma_color": 90.0,
                    "bilateral_sigma_space": 90.0,
                    "clahe_clip_limit": 2.8,
                    "clahe_grid_size": 8,
                    "sharpen_amount": 0.42,
                    "palette_size": 0,
                    "dither": "floyd_steinberg",
                    "slic_segments": 620,
                    "slic_compactness": 20.0,
                    "stroke_spacing_px": 2,
                    "optimisation_passes": 0,
                    "target_metrics": {"ssim": 0.985, "lpips": 0.045},
                },
            },
        }

    def _apply_style_profile(self, style_key: str) -> None:
        profile = self.paint_styles.get(style_key)
        if not profile:
            return

        self.selected_style_key = style_key
        self.active_style_profile = profile
        self.active_pipeline_profile = dict(profile.get("pipeline", {}))
        slicer_profile = profile.get("slicer", {})
        self.slicer.apply_style_profile(slicer_profile)
        self._active_brush_overrides = {
            key: dict(value)
            for key, value in profile.get("brushes", {}).items()
        }
        self.slicer.apply_brush_overrides(self._active_brush_overrides)
        self._brush_tool_cache.clear()

    def _prepare_planning_source(
        self,
        image_source: str,
        pipeline_profile: Dict[str, Any],
    ) -> Tuple[Any, List[str]]:
        """Führt optional die High-End-Pipeline aus und liefert das Analyse-Image."""

        pipeline_profile = pipeline_profile or {}
        self._superres_forced_off_reason = None
        run_pipeline = bool(
            pipeline_profile.get("run_pipeline")
            or pipeline_profile.get("force_full_process")
            or pipeline_profile.get("enable_superres")
        )

        if not run_pipeline:
            self.last_pipeline_result = None
            self.last_pipeline_summary = []
            self.pipeline_stroke_plan_mm = []
            return image_source, []

        allowed_keys = {
            "enable_superres",
            "superres_scale",
            "superres_model_path",
            "bilateral_diameter",
            "bilateral_sigma_color",
            "bilateral_sigma_space",
            "guided_radius",
            "guided_eps",
            "apply_guided_filter",
            "clahe_clip_limit",
            "clahe_grid_size",
            "sharpen_amount",
            "calibration_profile",
            "palette_size",
            "palette_colors",
            "dither",
            "slic_segments",
            "slic_compactness",
            "stroke_spacing_px",
            "target_metrics",
            "optimisation_passes",
        }

        pipeline_kwargs: Dict[str, Any] = {}
        for key in allowed_keys:
            if key in pipeline_profile and pipeline_profile[key] is not None:
                pipeline_kwargs[key] = pipeline_profile[key]

        if pipeline_kwargs.get("enable_superres"):
            model_path = pipeline_kwargs.get("superres_model_path") or os.environ.get("PAINTER_REAL_ESRGAN_MODEL")
            if not self.pipeline.super_resolution_available:
                LOGGER.info("Super-Resolution deaktiviert: Real-ESRGAN nicht verfügbar.")
                pipeline_kwargs["enable_superres"] = False
                self._superres_forced_off_reason = (
                    "Super-Resolution deaktiviert: Real-ESRGAN-Bibliothek nicht installiert."
                )
            elif not model_path:
                LOGGER.info("Super-Resolution deaktiviert: Kein Real-ESRGAN-Modellpfad konfiguriert.")
                pipeline_kwargs["enable_superres"] = False
                self._superres_forced_off_reason = (
                    "Super-Resolution deaktiviert: Es wurde kein Real-ESRGAN-Modellpfad konfiguriert."
                )
            elif model_path and not os.path.exists(model_path):
                LOGGER.info(
                    "Super-Resolution deaktiviert: Real-ESRGAN-Gewichte fehlen (Pfad: %s)",
                    model_path,
                )
                pipeline_kwargs["enable_superres"] = False
                self._superres_forced_off_reason = (
                    "Super-Resolution deaktiviert: Real-ESRGAN-Modellgewichte wurden nicht gefunden."
                )

        if "optimisation_passes" in pipeline_kwargs:
            try:
                pipeline_kwargs["optimisation_passes"] = int(pipeline_kwargs["optimisation_passes"])
            except (TypeError, ValueError):
                pipeline_kwargs.pop("optimisation_passes", None)

        result = self.pipeline.process(image_source, **pipeline_kwargs)
        self.last_pipeline_result = result

        summary = self._make_pipeline_summary(result, pipeline_profile)
        self.last_pipeline_summary = summary

        processed_rgb = (
            np.asarray(result.post_processed_rgb, dtype=np.float32)
            if getattr(result, "post_processed_rgb", None) is not None
            else np.empty((0, 0, 3), dtype=np.float32)
        )
        if processed_rgb.size == 0 and getattr(result, "calibrated_rgb", None) is not None:
            processed_rgb = np.asarray(result.calibrated_rgb, dtype=np.float32)
        if processed_rgb.size == 0:
            processed_rgb = self.analyzer._ensure_rgb01(image_source)
        if processed_rgb.size and float(processed_rgb.max()) > 1.0:
            processed_rgb = processed_rgb / 255.0

        base_rgb = self.analyzer._ensure_rgb01(image_source)
        base_h, base_w = base_rgb.shape[:2]

        if processed_rgb.shape[:2] != (base_h, base_w):
            processed_rgb = cv2.resize(
                processed_rgb,
                (base_w, base_h),
                interpolation=cv2.INTER_CUBIC,
            )

        self._build_pipeline_timeline(result, base_w, base_h)

        return processed_rgb, summary

    def _make_pipeline_summary(
        self,
        result: PipelineResult,
        pipeline_profile: Dict[str, Any],
    ) -> List[str]:
        lines: List[str] = [
            "Original-Stil: Vollständige Bildverarbeitungs-, Mal- und Postprozess-Pipeline aktiv."
        ]

        if self._superres_forced_off_reason:
            lines.append(self._superres_forced_off_reason)

        if result.config.get("enable_superres"):
            scale = result.config.get("superres_scale", 1)
            lines.append(f"Super-Resolution aktiv (Skalierung {scale}x).")

        palette_colors = 0
        dither_method = result.config.get("dither", "-")
        if result.palette and getattr(result.palette, "palette_rgb", None) is not None:
            palette_colors = int(result.palette.palette_rgb.shape[0])
        lines.append(
            f"Palette: {palette_colors} Farben | Dither: {dither_method} | Stroke-Abstand px: "
            f"{pipeline_profile.get('stroke_spacing_px', 'auto')}"
        )

        post_passes = result.config.get("post_process_passes")
        if isinstance(post_passes, int) and post_passes > 0:
            lines.append(f"Post-Processing: {post_passes}x durchlaufen für maximale Veredelung.")

        metrics = result.metrics or {}
        ssim = metrics.get("ssim")
        if isinstance(ssim, (int, float)) and not np.isnan(ssim):
            lines.append(f"Qualität (SSIM): {ssim:.4f}")
        lpips_val = metrics.get("lpips")
        if isinstance(lpips_val, (int, float)) and not np.isnan(lpips_val):
            lines.append(f"Perzeptuelle Abweichung (LPIPS): {lpips_val:.4f}")

        stroke_count = len(result.stroke_plan) if result.stroke_plan else 0
        lines.append(f"Generierte Stroke-Instruktionen: {stroke_count}")

        return lines

    def _build_pipeline_timeline(
        self,
        result: PipelineResult,
        base_w: int,
        base_h: int,
    ) -> None:
        self.pipeline_stroke_plan_mm = []
        if not result or not result.stroke_plan:
            return

        dithered = getattr(result, "dithered_rgb", None)
        if dithered is None or getattr(dithered, "shape", None) is None:
            return

        plan_h, plan_w = dithered.shape[:2]
        if plan_w == 0 or plan_h == 0:
            return

        scale_x = float(base_w) / float(plan_w)
        scale_y = float(base_h) / float(plan_h)

        converted: List[Dict[str, Any]] = []
        for instr in result.stroke_plan:
            path = getattr(instr, "path", None)
            if not path or len(path) < 2:
                continue

            mm_path: List[Tuple[float, float]] = []
            for x_px, y_px in path:
                orig_x = float(x_px) * scale_x
                orig_y = float(y_px) * scale_y
                X_mm = (orig_x / base_w) * self.slicer.work_w_mm
                Y_mm = (orig_y / base_h) * self.slicer.work_h_mm
                mm_path.append((X_mm, Y_mm))

            if len(mm_path) < 2:
                continue

            converted.append(
                {
                    "color_rgb": getattr(instr, "color_rgb", (0, 0, 0)),
                    "stage": getattr(instr, "stage", ""),
                    "technique": getattr(instr, "technique", ""),
                    "tool": getattr(instr, "tool", ""),
                    "shading": getattr(instr, "shading", ""),
                    "points": mm_path,
                }
            )

        self.pipeline_stroke_plan_mm = converted

    def _update_style_description(self) -> None:
        if not hasattr(self, "style_description_label"):
            return

        profile = self.paint_styles.get(self.selected_style_key, {})
        description = profile.get("description", "")
        analyzer_cfg = profile.get("analyzer", {})

        extras = []
        k_max = analyzer_cfg.get("k_max")
        if k_max:
            extras.append(f"bis zu {int(k_max)} Farbschichten")
        detail_boost = analyzer_cfg.get("detail_edge_boost")
        if detail_boost:
            extras.append(f"Detail-Boost ×{detail_boost:.2f}")
        if analyzer_cfg.get("preserve_edge_strokes"):
            extras.append("inkl. Kantennachzeichnung")

        summary = " · ".join(extras)
        text_parts = [description.strip()] if description else []
        if summary:
            text_parts.append(summary)

        self.style_description_label.setText("\n".join(part for part in text_parts if part))

    def _on_style_changed(self, style_key: str) -> None:
        if not style_key or style_key not in self.paint_styles:
            return

        self._apply_style_profile(style_key)
        self._update_style_description()

        if not self.current_image_path:
            return

        self.action_slice_plan()




    def _get_preview_canvas_dimensions(self) -> Tuple[int, int]:
        """Return the target canvas size for previews in original image dimensions."""

        fallback = (800, 800)

        if self._current_image_size:
            width, height = self._current_image_size
            if width > 0 and height > 0:
                return width, height

        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            if not pixmap.isNull():
                self._current_image_size = (pixmap.width(), pixmap.height())
                return self._current_image_size

        return fallback

    def _display_preview_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        """Show ``pixmap`` in the preview label while keeping aspect ratio."""

        if pixmap is None or pixmap.isNull():
            self.preview_label.setPixmap(QPixmap())
            self._last_preview_display = None
            return

        self._last_preview_display = QPixmap(pixmap)

        target_size = self.preview_label.size()
        if target_size.width() > 0 and target_size.height() > 0:
            scaled = pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
        else:
            self.preview_label.setPixmap(pixmap)

        self.preview_label.setText("")




    def _get_brush_parameters(
        self,
        tool_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        opacity_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        params = dict(self.slicer.get_brush_settings(tool_name))
        if overrides:
            params.update({k: v for k, v in overrides.items() if v is not None})
        if opacity_override is not None:
            params["opacity"] = opacity_override
        return params

    def _get_brush_tool(
        self,
        tool_name: str,
        *,
        brush_config: Optional[Dict[str, Any]] = None,
        opacity: Optional[float] = None,
    ) -> BrushTool:
        params = self._get_brush_parameters(tool_name, brush_config, opacity)
        cache_key = (tool_name, tuple(sorted(params.items())))
        brush = self._brush_tool_cache.get(cache_key)
        if brush is not None:
            return brush

        brush = BrushTool(
            width_px=params.get("width_px", 24.0),
            opacity=params.get("opacity", 0.85),
            edge_softness=params.get("edge_softness", 0.5),
            flow=params.get("flow", 0.7),
            spacing_px=params.get("spacing_px"),
        )
        self._brush_tool_cache[cache_key] = brush
        return brush

    def _mm_to_canvas_px(
        self,
        point: Tuple[float, float],
        canvas_w: int,
        canvas_h: int,
        work_w: float,
        work_h: float,
    ) -> Tuple[float, float]:
        x_mm, y_mm = point
        x_px = (x_mm / work_w) * canvas_w
        y_px = (y_mm / work_h) * canvas_h
        return (x_px, y_px)

    def _stroke_points_to_px(
        self,
        stroke: Dict[str, Any],
        canvas_w: int,
        canvas_h: int,
        work_w: float,
        work_h: float,
    ) -> List[Tuple[float, float]]:
        pts = stroke.get("points", [])
        return [self._mm_to_canvas_px(pt, canvas_w, canvas_h, work_w, work_h) for pt in pts]

    def _paint_stroke_with_limit(
        self,
        canvas: np.ndarray,
        stroke: Dict[str, Any],
        canvas_w: int,
        canvas_h: int,
        work_w: float,
        work_h: float,
        segments_completed: Optional[int] = None,
    ) -> None:
        points_px = self._stroke_points_to_px(stroke, canvas_w, canvas_h, work_w, work_h)
        if not points_px:
            return

        color_rgb = tuple(stroke.get("color_rgb", (255, 255, 255)))
        tool_name = str(stroke.get("tool", "medium_brush"))
        brush = self._get_brush_tool(
            tool_name,
            brush_config=stroke.get("brush"),
            opacity=stroke.get("opacity"),
        )

        total_segments = max(len(points_px) - 1, 0)
        if segments_completed is None or segments_completed >= total_segments:
            brush.render_path(canvas, points_px, color_rgb)
            return

        segments_completed = max(0, int(segments_completed))
        if segments_completed == 0:
            brush.stamp(canvas, points_px[0], color_rgb)
            return

        partial_points = points_px[: segments_completed + 1]
        brush.render_path(canvas, partial_points, color_rgb)

    def _render_strokes_to_array(
        self,
        upto_stroke: Optional[int],
        partial_segments: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        canvas_w, canvas_h = self._get_preview_canvas_dimensions()
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)

        strokes = self.paint_strokes_timeline
        if not strokes:
            return canvas

        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)

        partial_segments = partial_segments or {}
        max_index = len(strokes) - 1 if upto_stroke is None else upto_stroke

        for idx, stroke in enumerate(strokes):
            if idx > max_index and idx not in partial_segments:
                break

            segments_completed = partial_segments.get(idx)
            self._paint_stroke_with_limit(
                canvas,
                stroke,
                canvas_w,
                canvas_h,
                work_w,
                work_h,
                segments_completed=segments_completed,
            )

        return canvas

    def _array_to_qpixmap(self, canvas: np.ndarray) -> QPixmap:
        h, w, _ = canvas.shape
        clamped = np.clip(canvas, 0.0, 1.0)
        data = (clamped * 255).astype(np.uint8)
        image = QImage(data.data, w, h, data.strides[0], QImage.Format_RGBA8888)
        image = image.copy()
        return QPixmap.fromImage(image)

    def render_preview_full_colored(self):
        """
        Malt die komplette Szene farbig:
        - Jeder Stroke in seiner RGB-Farbe.
        - Keine Animation, alles fertig.
        """
        self.stop_preview_animation()

        if not self.paint_strokes_timeline or len(self.paint_strokes_timeline) == 0:
            self._display_preview_pixmap(None)
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")
            return

        canvas = self._render_strokes_to_array(upto_stroke=None)
        pm = self._array_to_qpixmap(canvas)

        self.preview_canvas_pixmap = pm
        self.current_highlight_segment = None
        self._display_preview_pixmap(pm)






    def render_preview_frame(self):
        """
        Rendert NICHT alle Pfade fertig,
        sondern nur bis zum aktuellen Animationsfortschritt:
        - Alle kompletten Strokes < anim_stroke_index
        - Angefangener Stroke bei anim_stroke_index bis anim_point_index
        """
        if not self.last_mm_paths or len(self.last_mm_paths) == 0:
            self._display_preview_pixmap(None)
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")
            return

        canvas_w, canvas_h = self._get_preview_canvas_dimensions()

        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
        brush = self._get_brush_tool(self.selected_tool)

        done_color = (200, 200, 200)
        active_color = (255, 240, 0)

        def path_to_px(path: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
            return [self._mm_to_canvas_px(pt, canvas_w, canvas_h, work_w, work_h) for pt in path]

        for s_idx in range(min(self.anim_stroke_index, len(self.last_mm_paths))):
            path = self.last_mm_paths[s_idx]
            pts_px = path_to_px(path)
            if pts_px:
                brush.render_path(canvas, pts_px, done_color)

        if self.anim_stroke_index < len(self.last_mm_paths):
            path = self.last_mm_paths[self.anim_stroke_index]
            pts_px = path_to_px(path)
            if pts_px:
                segments = min(self.anim_point_index, max(len(pts_px) - 1, 0))
                if segments <= 0:
                    brush.stamp(canvas, pts_px[0], active_color)
                else:
                    brush.render_path(canvas, pts_px[: segments + 1], active_color)

        pm = self._array_to_qpixmap(canvas)
        self.preview_canvas_pixmap = pm
        self._display_preview_pixmap(pm)

    def start_preview_animation(self):
        strokes = self.paint_strokes_timeline
        if not strokes:
            QMessageBox.warning(self, "Keine Pfade", "Bitte zuerst 'Slice planen'.")
            return

        # Toggle: läuft -> pausieren
        if self.anim_in_progress:
            self.anim_in_progress = False
            self.btn_play.setText("Play")
            self.anim_timer.stop()
            self._update_progress_ui()
            return

        # Start/Resume
        self.anim_in_progress = True
        self.btn_play.setText("Pause")

        # falls am Ende -> von vorne
        if self.anim_stroke_index >= len(strokes):
            self.anim_stroke_index = 0
            self.anim_point_index = 0

        self.anim_timer.stop()

        # Vor dem Fortsetzen aktuell markiertes Segment finalisieren
        self._finalize_current_highlight_to_canvas()
        self.current_highlight_segment = None

        # Basisbild (alle fertigen Strokes + bereits gemalte Segmente) aufbauen
        self._reset_preview_canvas_for_animation()

        # Geschwindigkeit aus Slider übernehmen
        self._update_animation_speed(self.speed_slider.value() if hasattr(self, "speed_slider") else 30)
        self.anim_timer.setSingleShot(False)  # Sicherheit: wiederholt feuern

        # Ersten Frame sofort anzeigen, damit direkt ein Highlight sichtbar ist
        self.animation_step()

        # Falls Animation noch aktiv -> Timer starten
        if self.anim_in_progress:
            self.anim_timer.start()



    def animation_step(self):
        strokes = self.paint_strokes_timeline
        if not self.anim_in_progress or not strokes:
            return

        # Zuvor hervorgehobenes Segment in Finalfarbe übernehmen
        if self.current_highlight_segment is not None:
            self._finalize_current_highlight_to_canvas()

        if self.preview_canvas_pixmap is None:
            self._reset_preview_canvas_for_animation()

        # Sicherstellen, dass wir auf einem gültigen Stroke stehen
        while self.anim_stroke_index < len(strokes):
            stroke = strokes[self.anim_stroke_index]
            pts = stroke.get("points", [])
            num_segments = max(len(pts) - 1, 0)

            if num_segments <= 0:
                self.anim_stroke_index += 1
                self.anim_point_index = 0
                continue

            if self.anim_point_index >= num_segments:
                self.anim_stroke_index += 1
                self.anim_point_index = 0
                continue

            break

        # Sind wir komplett fertig?
        if self.anim_stroke_index >= len(strokes):
            self._finalize_animation_display()
            return

        # Aktuelles Segment markieren und anzeigen
        segment_idx = self.anim_point_index
        self.current_highlight_segment = (self.anim_stroke_index, segment_idx)
        self.render_live_state()

        # Für den nächsten Tick ist das nächste Segment an der Reihe
        self.anim_point_index += 1
        self._update_progress_ui()




    def stop_preview_animation(self):
        self.anim_in_progress = False
        self.btn_play.setText("Play")
        self.anim_timer.stop()
        self._finalize_current_highlight_to_canvas()

        if self.preview_canvas_pixmap is not None:
            self._display_preview_pixmap(QPixmap(self.preview_canvas_pixmap))

        self._update_progress_ui()





    def _update_progress_ui(self):
        strokes = self.paint_strokes_timeline
        total_strokes = len(strokes)
        slider_max = max(total_strokes - 1, 0)

        # Slider darf nie größer sein als letzter Stroke
        if hasattr(self, "progress_slider"):
            previous_block = self.progress_slider.blockSignals(True)
            try:
                self.progress_slider.setMaximum(slider_max)
                if total_strokes == 0:
                    self.progress_slider.setValue(0)
                else:
                    self.progress_slider.setValue(min(self.anim_stroke_index, slider_max))
            finally:
                self.progress_slider.blockSignals(previous_block)

        if hasattr(self, "progress_label"):
            completed = min(self.anim_stroke_index, total_strokes)
            self.progress_label.setText(f"{completed} / {total_strokes}")








    def _rebuild_base_canvas_from_progress(self):
        """
        Baut self.preview_canvas_pixmap neu aus allen Strokes,
        die komplett abgeschlossen sind, also Index < anim_stroke_index.
        """
        base_index = self.anim_stroke_index - 1  # letzter vollständig fertiger Stroke
        pm = self._render_full_state_at(base_index)
        if pm is None:
            # fallback leere schwarze Fläche
            from PySide6.QtGui import QPixmap, QColor
            canvas_w, canvas_h = self._get_preview_canvas_dimensions()
            pm = QPixmap(canvas_w, canvas_h)
            pm.fill(QColor(0,0,0))
        self.preview_canvas_pixmap = pm




    def render_live_state(self):
        """
        Zeigt:
        - Alle fertigen Strokes (bis anim_stroke_index-1) in echter Farbe
        - Den aktuellen Stroke (anim_stroke_index) bis anim_point_index in NEON GRÜN
        """
        strokes = self.paint_strokes_timeline
        if not strokes:
            self._display_preview_pixmap(None)
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")
            return

        partial: Dict[int, int] = {}
        if 0 <= self.anim_stroke_index < len(strokes):
            partial[self.anim_stroke_index] = self.anim_point_index

        base_canvas = self._render_strokes_to_array(self.anim_stroke_index - 1, partial)
        self.preview_canvas_pixmap = self._array_to_qpixmap(base_canvas)

        frame_canvas = base_canvas.copy()
        highlight = self.current_highlight_segment
        if highlight is not None:
            stroke_idx, segment_idx = highlight
            if 0 <= stroke_idx < len(strokes):
                stroke = strokes[stroke_idx]
                pts_px = self._stroke_points_to_px(
                    stroke,
                    frame_canvas.shape[1],
                    frame_canvas.shape[0],
                    float(self.slicer.work_w_mm),
                    float(self.slicer.work_h_mm),
                )
                if len(pts_px) >= 2 and 0 <= segment_idx < len(pts_px) - 1:
                    base_tool = self._get_brush_tool(str(stroke.get("tool", "medium_brush")))
                    highlight_tool = BrushTool(
                        width_px=base_tool.width_px,
                        opacity=1.0,
                        edge_softness=max(0.1, base_tool.edge_softness * 0.6),
                        flow=1.0,
                        spacing_px=base_tool.spacing_px,
                    )
                    segment_points = pts_px[segment_idx : segment_idx + 2]
                    highlight_tool.render_path(frame_canvas, segment_points, (64, 255, 128))

        frame_pm = self._array_to_qpixmap(frame_canvas)
        self._display_preview_pixmap(frame_pm)








    def _render_full_state_at(self, stroke_index: int):
        """
        Baut ein Bild so, als wären alle Strokes bis einschließlich stroke_index
        vollständig gemalt (echte Farbe). Kein Grün.
        """
        strokes = self.paint_strokes_timeline
        if not strokes:
            return None

        last_idx = min(stroke_index, len(strokes) - 1)
        if last_idx < 0:
            canvas = self._render_strokes_to_array(upto_stroke=-1)
            return self._array_to_qpixmap(canvas)

        canvas = self._render_strokes_to_array(upto_stroke=last_idx)
        return self._array_to_qpixmap(canvas)



    def scrub_preview_to(self, value: int):
        strokes = self.paint_strokes_timeline
        if not strokes:
            return

        # Scrub pausiert immer
        self.anim_in_progress = False
        self.btn_play.setText("Play")
        self.anim_timer.stop()

        max_idx = len(strokes) - 1
        clamped = max(0, min(value, max_idx))

        # Alles bis clamped ist fertig -> nächster wäre clamped+1
        self.anim_stroke_index = clamped + 1
        self.anim_point_index = 0

        pm = self._render_full_state_at(clamped)
        if pm:
            self.preview_canvas_pixmap = pm
            self.current_highlight_segment = None
            self._display_preview_pixmap(pm)

        if self.progress_slider.value() != clamped:
            previous_block = self.progress_slider.blockSignals(True)
            try:
                self.progress_slider.setValue(clamped)
            finally:
                self.progress_slider.blockSignals(previous_block)
        self._update_progress_ui()




    def _reset_preview_canvas_for_animation(self):
        """
        Baut ein neues leeres Canvas auf (schwarzer Hintergrund)
        und malt alle Strokes VOR dem aktuellen anim_stroke_index vollständig drauf.
        Zusätzlich werden bereits abgeschlossene Segmente des aktuellen Strokes
        (bis anim_point_index) in Finalfarbe gerendert, damit beim Fortsetzen
        nahtlos angeschlossen wird.
        Danach benutzen wir dieses Canvas weiter inkrementell.
        """
        strokes = getattr(self, "paint_strokes_timeline", [])

        partial: Dict[int, int] = {}
        if 0 <= self.anim_stroke_index < len(strokes):
            partial[self.anim_stroke_index] = self.anim_point_index

        canvas = self._render_strokes_to_array(self.anim_stroke_index - 1, partial)
        self.preview_canvas_pixmap = self._array_to_qpixmap(canvas)


    def _finalize_segment_into_canvas(self, stroke_index: int, segment_index: int) -> None:
        strokes = self.paint_strokes_timeline
        if not strokes:
            return

        if stroke_index < 0 or stroke_index >= len(strokes):
            return

        stroke = strokes[stroke_index]
        pts = stroke.get("points", [])
        if len(pts) < 2 or segment_index < 0 or segment_index >= len(pts) - 1:
            return

        partial = {stroke_index: segment_index + 1}
        canvas = self._render_strokes_to_array(stroke_index - 1, partial)
        self.preview_canvas_pixmap = self._array_to_qpixmap(canvas)


    def _finalize_current_highlight_to_canvas(self) -> None:
        if not self.current_highlight_segment:
            return

        stroke_idx, segment_idx = self.current_highlight_segment
        self._finalize_segment_into_canvas(stroke_idx, segment_idx)
        self.current_highlight_segment = None


    def _finalize_animation_display(self) -> None:
        self.anim_timer.stop()
        self.anim_in_progress = False
        self.btn_play.setText("Play")

        final_pm = None
        if self.preview_canvas_pixmap is not None:
            final_pm = QPixmap(self.preview_canvas_pixmap)
        elif self.paint_strokes_timeline:
            final_pm = self._render_full_state_at(len(self.paint_strokes_timeline) - 1)

        if final_pm is not None:
            self._display_preview_pixmap(final_pm)
        else:
            self._display_preview_pixmap(None)
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")

        self.current_highlight_segment = None
        self._update_progress_ui()


    def _update_animation_speed(self, value: int):
        if not hasattr(self, "anim_timer"):
            return
        slider = getattr(self, "speed_slider", None)
        if slider is not None:
            min_val = float(slider.minimum())
            max_val = float(slider.maximum())
        else:
            min_val = 1.0
            max_val = 200.0

        span = max(max_val - min_val, 1.0)
        normalized = (float(value) - min_val) / span
        normalized = max(0.0, min(normalized, 1.0))

        min_interval = 5.0   # sehr schnell
        max_interval = 2000.0  # sehr langsam
        interval_ms = int(min_interval * ((max_interval / min_interval) ** normalized))
        interval_ms = max(1, interval_ms)

        self.anim_timer.setInterval(interval_ms)

