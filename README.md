# da_Vinci
"When the steel brush touches the soul of colors, the machine paints what humans dare to think."

## Photorealistic Underpainting Translation (New)

This repository now contains a complete, fully-commented PyTorch project that
teaches a painting robot how to convert layered grayscale underpaintings into
photorealistic RGB references. The implementation lives in the repository root
and is composed of the following modules:

* `config.py` – central place for hyperparameters and file-system paths.
* `dataset.py` – loads paired underpainting/target images and computes feature maps.
* `model.py` – a U-Net style generator with residual bottleneck.
* `losses.py` – composite loss combining reconstruction, perceptual and edge terms.
* `train.py` – end-to-end training loop with checkpointing and visual logging.
* `inference.py` – command line entry point for generating photorealistic images.
* `utils.py` – helper utilities for feature extraction and image handling.

### Expected data layout

```
data/
  train/
    underpaint/
    target/
  val/
    underpaint/
    target/
```

Underpainting and target images must share identical filenames so that the
dataset can pair them automatically.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows: PyTorch lässt sich nicht importieren (WinError 1114)

Falls beim Starten eines Skripts der Fehler

```
OSError: [WinError 1114] Eine DLL-Initialisierungsroutine ist fehlgeschlagen
```

auftritt, fehlen dem System in der Regel die nativen Bibliotheken von PyTorch.
Installiere zunächst die aktuelle **Microsoft Visual C++ 2015–2022
Redistributable (x64)**. Anschliessend kannst du die CPU-Version von PyTorch
explizit installieren:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Starte danach dein virtuelles Environment neu. Unsere Module prüfen den Import
von PyTorch beim Programmstart und geben andernfalls eine ausführliche
Anleitung aus.

### Training

```bash
python train.py
```

Model checkpoints are written to `checkpoints/` and qualitative samples to
`results/`.

### Inference

```bash
python inference.py --input path/to/underpainting.png --output path/to/output.png --checkpoint checkpoints/epoch_0100.pth
```

The helper function `generate_photo_from_underpainting` is also available for
integration into larger control stacks.

## Verbesserte Bildverarbeitung

Die Bildanalyse verwendet jetzt eine mehrstufige High-End-Pipeline, die auf den in
[Neptune.ai beschriebenen Best Practices](https://neptune.ai/blog/image-processing-python-libraries-for-machine-learning)
aufbaut. Über Pillow, SciPy und SimpleITK werden Farbdynamik, anisotrope
Rauschunterdrückung sowie detailerhaltende Schärfung kombiniert, bevor OpenCV die
Segmentierung übernimmt. Das Resultat sind deutlich glattere Farbflächen und klarere
Konturen für den Malplaner.

Alle benötigten Python-Abhängigkeiten, inklusive der neuen Bibliotheken (SciPy,
SimpleITK, Pillow), finden sich in `requirements.txt`.

### Layer-orientierte API

Für Workflows, die einzelne Ebenen separat weiterverarbeiten möchten, stehen nun
folgende High-Level-Helfer bereit:

* `painterslicer.image_analysis.segment_image_into_layers(image_source)` liefert
  neben den Masken für Hintergrund, Mittelgrund und Details auch direkt die
  RGB-Layer (Float 0‒1), bei denen Off-Mask-Bereiche transparent gesetzt sind.
* `painterslicer.image_analysis.enhance_layer(layer_rgb01, mask, scale, model_path=None)`
  nutzt intern Real-ESRGAN (falls verfügbar), um einen einzelnen Layer
  hochzuskalieren und wendet anschließend wieder die Maske an, sodass keine
  Ausfransungen außerhalb der Maske entstehen.
* `painterslicer.image_analysis.compose_layers(background, midground, foreground, feather_radius=3)`
  setzt drei RGBA-Layer in der Reihenfolge Hintergrund → Mittelgrund → Vordergrund
  per Alpha-Blending zusammen und erlaubt optionales Feathering der Masken über
  einen Gauß-Blur.

Alle Funktionen arbeiten rein auf NumPy-Arrays und lassen sich somit auch in Tests
oder externen Tools unkompliziert einsetzen.

## Übernahme von Änderungen aus einem externen Branch

Falls du die hier bereitgestellten Commits in dein lokales Projekt übernehmen möchtest, führe die folgenden Schritte im Stammverzeichnis deines PainterSlicer-Git-Repositories aus (dort, wo sich der Ordner `.git` befindet). Wenn du dich nicht im richtigen Verzeichnis befindest, meldet Git wie in deinem Screenshot, dass es sich „nicht um ein Git-Repository“ handelt.

1. **In das Projektverzeichnis wechseln**  
   ```bash
   cd /pfad/zu/deinem/projekt
   ```

2. **Remote für dieses Repository hinzufügen** (nur einmal erforderlich):  
   ```bash
   git remote add codex <URL-zu-diesem-Repository>
   ```

3. **Branch abrufen**:  
   ```bash
   git fetch codex work
   ```

4. **Ziel-Branch auschecken** (zum Beispiel `main`):  
   ```bash
   git checkout main
   ```

5. **Änderungen übernehmen**:  
   ```bash
   git merge codex/work
   ```
   oder, wenn du nur einen einzelnen Commit benötigst:  
   ```bash
   git cherry-pick 11cc2e94c9caef1665b39961ec0551bec7198268
   ```

Sollte Git Konflikte melden, öffne die betroffenen Dateien, löse die Konflikte, füge die Dateien hinzu (`git add <Datei>`) und setze den Merge beziehungsweise Cherry-Pick fort (`git merge --continue` oder `git cherry-pick --continue`). Abschließend kannst du deine Änderungen wie gewohnt mit `git push` in dein Remote-Repository übertragen.
