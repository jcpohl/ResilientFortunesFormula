from __future__ import annotations
from pathlib import Path
import os, yaml, re
from typing import Dict, Optional

# YYYY-MM-DD   or   YYYYMMDD
_date_rx = re.compile(r"^\d{4}(?:-?\d{2}-?\d{2})$")

def load_cfg() -> Dict:
    fp = Path(os.getenv("PIPELINE_CFG", "pipeline_config.yaml")).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"pipeline_config.yaml not found at {fp}")
    return yaml.safe_load(fp.read_text("utf-8")) or {}

def resolve_run_dir(
    swan_year: str | int | None = None,
    must_have: str | None = None,
    run_tag: str | None = None,
) -> Path:
    """
    Locate the run folder for *swan_year*.

    Priority order
      1. $RUN_DIR        – explicit absolute path
      2. run_tag arg     – function argument
      3. $RUN_TAG env    – usually coming from the orchestrator
      4. $RUN_DATE env   – legacy override
      5. newest folder that already contains *must_have*
      6. newest dated folder in event=<YEAR>/
    """
    cfg     = load_cfg()
    events  = {str(k): v for k, v in cfg.get("events", {}).items()}
    sy      = str(swan_year or os.getenv("SWAN_YEAR") or next(iter(events)))
    if sy not in events:
        raise KeyError(f"SWAN_YEAR={sy} not in YAML events:")

    out_root  = Path(cfg["defaults"]["OUTPUT_ROOT"]).expanduser()
    event_dir = out_root / f"event={sy}"

    # 1 ─ explicit full path
    if os.getenv("RUN_DIR"):
        return Path(os.getenv("RUN_DIR")).expanduser()

    # 2/3/4 ─ explicit tag
    run_tag = (run_tag
               or os.getenv("RUN_TAG")
               or os.getenv("RUN_DATE"))
    if run_tag:
        run_tag = run_tag.replace("--", "-")
        cand = event_dir / run_tag
        if cand.is_dir():
            if must_have and not (cand / must_have).exists():
                raise FileNotFoundError(f"{must_have} not found in {cand}")
            return cand
        raise FileNotFoundError(f"Run-tag {run_tag} not found for event {sy}")

    # 5 ─ newest folder that already contains *must_have*
    if must_have:
        hits = sorted(event_dir.glob(f"*/{must_have}"))
        if hits:
            return hits[-1].parents[1]
        raise FileNotFoundError(f"No run contains {must_have} in {event_dir}")

    # 6 ─ newest dated folder
    dated = [p for p in event_dir.iterdir() if p.is_dir() and _date_rx.match(p.name)]
    if not dated:
        raise FileNotFoundError(f"No run folders found in {event_dir}")
    return sorted(dated)[-1]