import json
import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_machine_config() -> Dict[str, Any]:
    path = os.path.join(BASE_DIR, "machine_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_brush_presets() -> Dict[str, Any]:
    path = os.path.join(BASE_DIR, "brush_presets.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
