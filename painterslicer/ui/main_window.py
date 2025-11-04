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
from PySide6.QtGui import QIcon, QAction, QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QTimer

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from painterslicer.image_analysis.analyzer import ImageAnalyzer
from painterslicer.image_analysis.pipeline import PaintingPipeline, PipelineResult
from painterslicer.slicer.slicer_core import PainterSlicer


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
        self._apply_style_profile(self.selected_style_key)

        self.last_pipeline_result: Optional[PipelineResult] = None
        self.last_pipeline_summary: List[str] = []
        self.pipeline_stroke_plan_mm: List[Dict[str, Any]] = []

        # Animation zurücksetzen
        # --- Animation / Preview State ---
        from PySide6.QtCore import QTimer, Qt

#        self.speed_slider.valueChanged.connect(self._update_animation_speed)

        # ...
        self.anim_in_progress = False
        self.anim_stroke_index = 0
        self.anim_point_index = 0
        self.paint_strokes_timeline = []
        self.preview_canvas_pixmap = None
        self.last_planning_result = None

        # QTimer EINMAL erstellen und dauerhaft behalten
        self.anim_timer = QTimer(self)
        self.anim_timer.setTimerType(Qt.PreciseTimer)  # präziser Takt
        self.anim_timer.setSingleShot(False)  # WICHTIG: NICHT SingleShot
        self.anim_timer.setInterval(30)  # Default, wird bei Play gesetzt
        #self.anim_timer.timeout.connect(self.animation_step)
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
        normalized_layers = self.slicer.normalize_color_layers_to_mm(
            self.current_image_path,
            color_layers
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

        clean_interval = slicer_style.get("clean_interval") if slicer_style else None

        paintcode_multi = self.slicer.multi_layer_paintcode(
            normalized_layers,
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
        for layer_info, layer in zip(color_layers, normalized_layers):
            rgb = layer["color_rgb"]
            stage = layer_info.get("stage")
            technique = layer_info.get("technique")
            label = layer_info.get("label")
            for stroke in layer["mm_paths"]:
                if len(stroke) < 2:
                    continue
                self.paint_strokes_timeline.append({
                    "color_rgb": rgb,
                    "points": stroke,  # Liste [(x_mm,y_mm), ...]
                    "stage": stage,
                    "technique": technique,
                    "label": label,
                })

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
        lines_out.append("\n--- Farb-Layer Info ---\n")
        lines_out.append(f"Anzahl Farblayer: {len(normalized_layers)}\n")

        for idx, (layer_info, layer) in enumerate(zip(color_layers, normalized_layers)):
            rgb = layer["color_rgb"]
            coverage_pct = float(layer_info.get("coverage", 0.0) * 100.0)
            stage = layer_info.get("stage", "?")
            order = layer_info.get("order", idx)
            tool = layer_info.get("tool", "-")
            technique = layer_info.get("technique", "-")
            path_count = len(layer["mm_paths"])
            lines_out.append(
                f"Layer {idx+1:02d} (Order {order}, Stage {stage}): RGB={rgb}, "
                f"Coverage={coverage_pct:.1f}%, Pfade={path_count}, Werkzeug={tool}, "
                f"Technik={technique}"
            )
            lines_out.append(
                "    detail={detail:.2f} mid={mid:.2f} background={bg:.2f} "
                "texture={tex:.2f} highlight={hi:.2f} shadow={sh:.2f} "
                "contrast={co:.2f} colorVar={cv:.2f}".format(
                    detail=float(layer_info.get("detail_ratio", 0.0)),
                    mid=float(layer_info.get("mid_ratio", 0.0)),
                    bg=float(layer_info.get("background_ratio", 0.0)),
                    tex=float(layer_info.get("texture_strength", 0.0)),
                    hi=float(layer_info.get("highlight_strength", 0.0)),
                    sh=float(layer_info.get("shadow_strength", 0.0)),
                    co=float(layer_info.get("contrast_strength", 0.0)),
                    cv=float(layer_info.get("color_variance_strength", 0.0)),
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

        # Fortschritt-Slider setzen (0..len-1)
        if hasattr(self, "progress_slider"):
            self.progress_slider.setMinimum(0)
            self.progress_slider.setMaximum(max(len(self.paint_strokes_timeline) - 1, 0))
            self.progress_slider.setValue(0)

        # erstes Bild: noch nix gemalt
        pm0 = self._render_full_state_at(-1)  # liefert schwarze Fläche
        if pm0 is not None:
            self.preview_label.setPixmap(pm0)
            self.preview_label.setText("")
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
                    "palette_size": 24,
                    "dither": "floyd_steinberg",
                    "slic_segments": 620,
                    "slic_compactness": 20.0,
                    "stroke_spacing_px": 2,
                    "optimisation_passes": 1,
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

    def _prepare_planning_source(
        self,
        image_source: str,
        pipeline_profile: Dict[str, Any],
    ) -> Tuple[Any, List[str]]:
        """Führt optional die High-End-Pipeline aus und liefert das Analyse-Image."""

        pipeline_profile = pipeline_profile or {}
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





    def render_preview_full_colored(self):
        """
        Malt die komplette Szene farbig:
        - Jeder Stroke in seiner RGB-Farbe.
        - Keine Animation, alles fertig.
        """
        from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
        from PySide6.QtCore import Qt

        if not self.paint_strokes_timeline or len(self.paint_strokes_timeline) == 0:
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")
            self.preview_label.setPixmap(QPixmap())
            return

        canvas_w = 800
        canvas_h = 800

        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)

        pm = QPixmap(canvas_w, canvas_h)
        pm.fill(QColor(0, 0, 0))

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)

        for stroke in self.paint_strokes_timeline:
            pts = stroke["points"]
            color_rgb = stroke["color_rgb"]
            if len(pts) < 2:
                continue

            pen = QPen(QColor(color_rgb[0], color_rgb[1], color_rgb[2]))
            pen.setWidth(2)
            painter.setPen(pen)

            for i in range(len(pts) - 1):
                (x1_mm, y1_mm) = pts[i]
                (x2_mm, y2_mm) = pts[i + 1]

                x1_px = int((x1_mm / work_w) * canvas_w)
                y1_px = int((y1_mm / work_h) * canvas_h)
                x2_px = int((x2_mm / work_w) * canvas_w)
                y2_px = int((y2_mm / work_h) * canvas_h)

                painter.drawLine(x1_px, y1_px, x2_px, y2_px)

        painter.end()

        self.preview_label.setPixmap(pm)
        self.preview_label.setText("")






    def render_preview_frame(self):
        """
        Rendert NICHT alle Pfade fertig,
        sondern nur bis zum aktuellen Animationsfortschritt:
        - Alle kompletten Strokes < anim_stroke_index
        - Angefangener Stroke bei anim_stroke_index bis anim_point_index
        """

        from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
        from PySide6.QtCore import Qt

        if not self.last_mm_paths or len(self.last_mm_paths) == 0:
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")
            self.preview_label.setPixmap(QPixmap())
            return

        canvas_w = 800
        canvas_h = 800

        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)

        pm = QPixmap(canvas_w, canvas_h)
        pm.fill(QColor(0, 0, 0))

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Strokes, die schon fertig gemalt sind → hellere Farbe
        done_pen = QPen(QColor(200, 200, 200))
        done_pen.setWidth(2)

        # Stroke, der gerade gemalt wird → hell/kontrast
        active_pen = QPen(QColor(255, 255, 0))
        active_pen.setWidth(2)

        # 1. Alle komplett fertigen Strokes zeichnen
        for s_idx in range(min(self.anim_stroke_index, len(self.last_mm_paths))):
            stroke = self.last_mm_paths[s_idx]
            if len(stroke) < 2:
                continue

            painter.setPen(done_pen)
            for i in range(len(stroke) - 1):
                (x1_mm, y1_mm) = stroke[i]
                (x2_mm, y2_mm) = stroke[i + 1]

                x1_px = int((x1_mm / work_w) * canvas_w)
                y1_px = int((y1_mm / work_h) * canvas_h)
                x2_px = int((x2_mm / work_w) * canvas_w)
                y2_px = int((y2_mm / work_h) * canvas_h)

                painter.drawLine(x1_px, y1_px, x2_px, y2_px)

        # 2. Den Stroke, der gerade „live“ gemalt wird
        if self.anim_stroke_index < len(self.last_mm_paths):
            stroke = self.last_mm_paths[self.anim_stroke_index]
            if len(stroke) >= 2:
                painter.setPen(active_pen)

                # wir zeichnen nur bis anim_point_index
                max_i = min(self.anim_point_index, len(stroke) - 1)
                for i in range(max_i):
                    (x1_mm, y1_mm) = stroke[i]
                    (x2_mm, y2_mm) = stroke[i + 1]

                    x1_px = int((x1_mm / work_w) * canvas_w)
                    y1_px = int((y1_mm / work_h) * canvas_h)
                    x2_px = int((x2_mm / work_w) * canvas_w)
                    y2_px = int((y2_mm / work_h) * canvas_h)

                    painter.drawLine(x1_px, y1_px, x2_px, y2_px)

        painter.end()

        self.preview_label.setPixmap(pm)
        self.preview_label.setText("")

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
            return

        # Start/Resume
        self.anim_in_progress = True
        self.btn_play.setText("Pause")

        # falls am Ende -> von vorne
        if self.anim_stroke_index >= len(strokes):
            self.anim_stroke_index = 0
            self.anim_point_index = 0

        # Geschwindigkeit aus Slider
        interval_ms = int(self.speed_slider.value()) if hasattr(self, "speed_slider") else 30
        if interval_ms < 5: interval_ms = 5
        self.anim_timer.setInterval(interval_ms)
        self.anim_timer.setSingleShot(False)  # Sicherheit: wiederholt feuern

        # Basisbild (alles vor current stroke) bauen und ersten Frame zeigen
        self._rebuild_base_canvas_from_progress()
        self.render_live_state()
        self._update_progress_ui()

        # Timer los
        self.anim_timer.start()



    def animation_step(self):
        strokes = self.paint_strokes_timeline
        if not self.anim_in_progress or not strokes:
            return

        # fertig?
        if self.anim_stroke_index >= len(strokes):
            final_pm = self._render_full_state_at(len(strokes) - 1)
            if final_pm:
                self.preview_label.setPixmap(final_pm)
                self.preview_label.setText("")
            self.anim_timer.stop()
            self.anim_in_progress = False
            self.btn_play.setText("Play")
            self._update_progress_ui()
            return

        pts = strokes[self.anim_stroke_index]["points"]

        if len(pts) < 2:
            self.anim_stroke_index += 1
            self.anim_point_index = 0
        else:
            self.anim_point_index += 1
            if self.anim_point_index >= len(pts) - 1:
                self.anim_stroke_index += 1
                self.anim_point_index = 0

        if self.anim_stroke_index >= len(strokes):
            final_pm = self._render_full_state_at(len(strokes) - 1)
            if final_pm:
                self.preview_label.setPixmap(final_pm)
                self.preview_label.setText("")
            self.anim_timer.stop()
            self.anim_in_progress = False
            self.btn_play.setText("Play")
            self._update_progress_ui()
            return

        self._rebuild_base_canvas_from_progress()
        self.render_live_state()
        self._update_progress_ui()




    def stop_preview_animation(self):
        self.anim_in_progress = False
        self.btn_play.setText("Play")
        self.anim_timer.stop()
        self._update_progress_ui()





    def _update_progress_ui(self):
        strokes = self.paint_strokes_timeline
        max_idx = max(len(strokes) - 1, 0)

        # Slider darf nie größer sein als letzter Stroke
        if hasattr(self, "progress_slider"):
            self.progress_slider.setMaximum(max_idx)
            # Während Play folgen wir automatisch der Animation
            if self.anim_in_progress:
                self.progress_slider.setValue(min(self.anim_stroke_index, max_idx))

        if hasattr(self, "progress_label"):
            self.progress_label.setText(f"{min(self.anim_stroke_index, max_idx)} / {max_idx}")








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
            pm = QPixmap(800, 800)
            pm.fill(QColor(0,0,0))
        self.preview_canvas_pixmap = pm




    def render_live_state(self):
        """
        Zeigt:
        - Alle fertigen Strokes (bis anim_stroke_index-1) in echter Farbe
        - Den aktuellen Stroke (anim_stroke_index) bis anim_point_index in NEON GRÜN
        """
        from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
        from PySide6.QtCore import Qt

        strokes = self.paint_strokes_timeline
        if not strokes:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Keine Pfade vorhanden.\nBitte 'Slice planen'.")
            return

        if self.preview_canvas_pixmap is None:
            self._rebuild_base_canvas_from_progress()

        frame_pm = QPixmap(self.preview_canvas_pixmap)

        # aktuellen Stroke (teilweise) grün highlighten
        if self.anim_stroke_index < len(strokes):
            stroke = strokes[self.anim_stroke_index]
            pts = stroke["points"]

            if len(pts) >= 2:
                painter = QPainter(frame_pm)
                painter.setRenderHint(QPainter.Antialiasing, True)

                neon_green = QColor(0, 255, 0, 255)
                pen = QPen(neon_green)
                pen.setWidth(3)
                painter.setPen(pen)

                # Wir zeichnen nur die ersten anim_point_index Segmente
                max_i = min(self.anim_point_index, len(pts) - 1)
                work_w = float(self.slicer.work_w_mm)
                work_h = float(self.slicer.work_h_mm)

                for i in range(max_i):
                    (x1_mm, y1_mm) = pts[i]
                    (x2_mm, y2_mm) = pts[i + 1]

                    x1_px = int((x1_mm / work_w) * 800)
                    y1_px = int((y1_mm / work_h) * 800)
                    x2_px = int((x2_mm / work_w) * 800)
                    y2_px = int((y2_mm / work_h) * 800)

                    painter.drawLine(x1_px, y1_px, x2_px, y2_px)

                painter.end()

        self.preview_label.setPixmap(frame_pm)
        self.preview_label.setText("")








    def _render_full_state_at(self, stroke_index: int):
        """
        Baut ein Bild so, als wären alle Strokes bis einschließlich stroke_index
        vollständig gemalt (echte Farbe). Kein Grün.
        """
        from PySide6.QtGui import QPixmap, QPainter, QPen, QColor

        strokes = self.paint_strokes_timeline
        if not strokes:
            return None

        canvas_w = 800
        canvas_h = 800
        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)

        pm = QPixmap(canvas_w, canvas_h)
        pm.fill(QColor(0, 0, 0))

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)

        last_idx = min(stroke_index, len(strokes) - 1)
        if last_idx < 0:
            # "nix gemalt"
            painter.end()
            return pm

        for s_idx in range(last_idx + 1):
            stroke = strokes[s_idx]
            pts = stroke["points"]
            rgb = stroke["color_rgb"]
            if len(pts) < 2:
                continue

            pen = QPen(QColor(rgb[0], rgb[1], rgb[2], 255))
            pen.setWidth(3)
            painter.setPen(pen)

            for i in range(len(pts) - 1):
                (x1_mm, y1_mm) = pts[i]
                (x2_mm, y2_mm) = pts[i + 1]

                x1_px = int((x1_mm / work_w) * canvas_w)
                y1_px = int((y1_mm / work_h) * canvas_h)
                x2_px = int((x2_mm / work_w) * canvas_w)
                y2_px = int((y2_mm / work_h) * canvas_h)

                painter.drawLine(x1_px, y1_px, x2_px, y2_px)

        painter.end()
        return pm



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
            self.preview_label.setPixmap(pm)
            self.preview_label.setText("")

        self.progress_slider.setValue(clamped)
        self._update_progress_ui()




    def _reset_preview_canvas_for_animation(self):
        """
        Baut ein neues leeres Canvas auf (schwarzer Hintergrund)
        und malt alle Strokes VOR dem aktuellen anim_stroke_index vollständig drauf.
        Danach benutzen wir dieses Canvas weiter inkrementell.
        """
        from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
        strokes = getattr(self, "paint_strokes_timeline", [])

        canvas_w = 800
        canvas_h = 800
        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)

        # Neues Pixmap als Basis
        pm = QPixmap(canvas_w, canvas_h)
        pm.fill(QColor(0, 0, 0))

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Male alle komplett fertigen Strokes schon rein (voller Deckkraft)
        for s_idx in range(min(self.anim_stroke_index, len(strokes))):
            pts = strokes[s_idx]["points"]
            rgb = strokes[s_idx]["color_rgb"]
            if len(pts) < 2:
                continue

            pen = QPen(QColor(rgb[0], rgb[1], rgb[2], 255))  # volle Deckkraft
            pen.setWidth(3)
            painter.setPen(pen)

            for i in range(len(pts) - 1):
                (x1_mm, y1_mm) = pts[i]
                (x2_mm, y2_mm) = pts[i + 1]
                x1_px = int((x1_mm / work_w) * canvas_w)
                y1_px = int((y1_mm / work_h) * canvas_h)
                x2_px = int((x2_mm / work_w) * canvas_w)
                y2_px = int((y2_mm / work_h) * canvas_h)
                painter.drawLine(x1_px, y1_px, x2_px, y2_px)

        painter.end()
        self.preview_canvas_pixmap = pm



    def _finalize_current_stroke_into_canvas(self):
        """
        Den gerade fertigen Stroke in echter Farbe dauerhaft ins Canvas 'einbrennen'.
        """
        from PySide6.QtGui import QPainter, QPen, QColor, QPixmap

        strokes = self.paint_strokes_timeline
        if self.preview_canvas_pixmap is None:
            self._reset_preview_canvas_for_animation()

        if self.anim_stroke_index >= len(strokes):
            return

        stroke = strokes[self.anim_stroke_index]
        pts = stroke["points"]
        color_rgb = stroke["color_rgb"]

        if len(pts) < 2:
            return

        canvas_w = self.preview_canvas_pixmap.width()
        canvas_h = self.preview_canvas_pixmap.height()
        work_w = float(self.slicer.work_w_mm)
        work_h = float(self.slicer.work_h_mm)

        # aktuelle Leinwand kopieren & darauf malen
        base_pm = QPixmap(self.preview_canvas_pixmap)
        p = QPainter(base_pm)
        p.setRenderHint(QPainter.Antialiasing, True)

        # final in voller Deckkraft, Tool-Strichbreite 3 als "Pinsel"
        pen = QPen(QColor(color_rgb[0], color_rgb[1], color_rgb[2], 255))
        pen.setWidth(3)
        p.setPen(pen)

        for i in range(len(pts) - 1):
            (x1_mm, y1_mm) = pts[i]
            (x2_mm, y2_mm) = pts[i + 1]

            x1_px = int((x1_mm / work_w) * canvas_w)
            y1_px = int((y1_mm / work_h) * canvas_h)
            x2_px = int((x2_mm / work_w) * canvas_w)
            y2_px = int((y2_mm / work_h) * canvas_h)

            p.drawLine(x1_px, y1_px, x2_px, y2_px)

        p.end()

        # neue Leinwand übernehmen
        self.preview_canvas_pixmap = base_pm


    def _update_animation_speed(self, value: int):
        if not hasattr(self, "anim_timer"):
            return
        interval_ms = int(value)
        if interval_ms < 5: interval_ms = 5
        self.anim_timer.setInterval(interval_ms)

