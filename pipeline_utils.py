from __future__ import annotations
from pathlib import Path
import os, yaml, re
from typing import Dict, List, Union, Optional

# YYYY-MM-DD   or   YYYYMMDD
_date_rx = re.compile(r"^\d{4}(?:-?\d{2}-?\d{2})$")

# ────────────────────────────────────────────────────────────────
#  CONFIG LOADER
# ────────────────────────────────────────────────────────────────
def load_cfg() -> Dict:
    """Read `pipeline_config.yaml` (path overridable via $PIPELINE_CFG)."""
    fp = Path(os.getenv("PIPELINE_CFG", "pipeline_config.yaml")).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(f"pipeline_config.yaml not found at {fp}")
    return yaml.safe_load(fp.read_text("utf-8")) or {}


# ────────────────────────────────────────────────────────────────
#  RUN-FOLDER RESOLVER (robust to list-style `must_have`)
# ────────────────────────────────────────────────────────────────
def resolve_run_dir(
    swan_year: str | int | None = None,
    must_have: Union[str, Path, List[Union[str, Path]], None] = None,
    run_tag: str | None = None,
) -> Path:
    """
    Locate the run folder for *swan_year*.

    Priority order
      1. $RUN_DIR                – explicit absolute path
      2. run_tag argument
      3. $RUN_TAG envvar
      4. $RUN_DATE envvar (legacy)
      5. most-recent folder that already contains *must_have*
      6. newest dated folder in event=<YEAR>/

    *must_have* may be:
        • a single relative path  (str / Path)
        • a list/tuple of paths   – *all* must exist inside the candidate
    """
    cfg     = load_cfg()
    events  = {str(k): v for k, v in cfg.get("events", {}).items()}

    sy = str(swan_year or os.getenv("SWAN_YEAR") or next(iter(events)))
    if sy not in events:
        raise KeyError(f"SWAN_YEAR={sy} not in YAML events")

    out_root  = Path(cfg["defaults"]["OUTPUT_ROOT"]).expanduser()
    event_dir = out_root / f"event={sy}"
    if not event_dir.is_dir():
        raise FileNotFoundError(f"{event_dir} not found")

    # ── normalise must_have to list ──────────────────────────────
    if must_have is None:
        must_have = []
    elif isinstance(must_have, (str, Path)):
        must_have = [must_have]

    def _satisfies(folder: Path) -> bool:
        """True iff *all* required artefacts exist inside *folder*."""
        return all((folder / req).exists() for req in must_have)

    # 1 ─ explicit absolute path via $RUN_DIR
    if os.getenv("RUN_DIR"):
        cand = Path(os.getenv("RUN_DIR")).expanduser()
        if _satisfies(cand):
            return cand
        raise FileNotFoundError(f"{cand} lacks required file(s): {must_have}")

    # 2/3/4 ─ explicit tag
    run_tag = run_tag or os.getenv("RUN_TAG") or os.getenv("RUN_DATE")
    if run_tag:
        cand = event_dir / run_tag.replace("--", "-")
        if cand.is_dir() and _satisfies(cand):
            return cand
        raise FileNotFoundError(f"{cand} missing or incomplete for {sy}")

    # 5 ─ newest folder that already contains *must_have*
    runs = sorted([p for p in event_dir.iterdir() if p.is_dir()])
    for folder in reversed(runs):
        if _satisfies(folder):
            return folder

    # 6 ─ newest dated folder
    dated = [p for p in runs if _date_rx.match(p.name)]
    if dated:
        return sorted(dated)[-1]

    raise FileNotFoundError(
        f"No run folder in {event_dir} satisfies requirement(s): {must_have}"
    )
