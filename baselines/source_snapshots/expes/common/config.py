"""YAML config helpers for expes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML required for yaml configs: pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(cfg)
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML required for yaml configs: pip install pyyaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
