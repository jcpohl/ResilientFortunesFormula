from __future__ import annotations
from pathlib import Path
import os, yaml, re
from typing import Dict

_date_rx = re.compile(r"^\d{8}$")

def load_cfg() -> Dict:
    fp = Path(os.getenv("PIPELINE_CFG", "pipeline_config.yaml")).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"pipeline_config.yaml not found at {fp}")
    # Use UTF-8 encoding to avoid decoding errors
    return yaml.safe_load(fp.read_text(encoding="utf-8")) or {}

def resolve_run_dir(swan_year: str | int = None, must_have: str | None = None) -> Path:
    cfg = load_cfg()
    events = {str(k): v for k, v in cfg.get("events", {}).items()}
    sy = str(swan_year or os.getenv("SWAN_YEAR") or next(iter(events)))
    if sy not in events:
        raise KeyError(f"SWAN_YEAR={sy} not in YAML events:")
    out_root = Path(cfg["defaults"]["OUTPUT_ROOT"]).expanduser()
    event_dir = out_root / f"event={sy}"

    if os.getenv("RUN_DIR"):
        return Path(os.getenv("RUN_DIR")).expanduser()
    if os.getenv("RUN_DATE"):
        return event_dir / os.getenv("RUN_DATE")

    # auto-discover newest run:
    if must_have:
        hits = list(event_dir.glob(f"*/{must_have}"))
        if not hits:
            raise FileNotFoundError(f"No run contains {must_have} in {event_dir}")
        return hits[-1].parents[1]      # …/<run-tag>/
    # fallback: newest dated folder
    cand = [p for p in event_dir.iterdir() if p.is_dir() and _date_rx.match(p.name)]
    if not cand:
        raise FileNotFoundError(f"No run folders found in {event_dir}")
    return sorted(cand)[-1]