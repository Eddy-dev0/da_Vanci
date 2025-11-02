# da_Vinci
"When the steel brush touches the soul of colors, the machine paints what humans dare to think."

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
