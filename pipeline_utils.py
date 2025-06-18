#!/usr/bin/env python
"""
Common helpers shared by all pipeline stages.
v2.1 · 2025-06-18
"""

from __future__ import annotations
import logging, os, re, sys
from pathlib import Path
from typing  import Dict, Optional

import yaml

# ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
CFG_PATH   = ROOT / "pipeline_config.yaml"
_OUTPUT_CACHE: dict[str, Path] = {}          # memo: SWAN_YEAR → run dir

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-7s | [utils] %(message)s",
    stream = sys.stdout,
)
ulog = logging.getLogger("pipeline_utils")

# ═════════════════════════ 1 · CONFIG ════════════════════════════
def load_cfg(force_reload: bool = False) -> Dict:
    """
    Return *pipeline_config.yaml* as a dict (cached).

    Tolerates either ``output_root`` **or** the legacy upper-case
    ``OUTPUT_ROOT`` in *defaults*.
    """
    if force_reload or not hasattr(load_cfg, "_cache"):
        with open(CFG_PATH, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        # ----- normalise OUTPUT_ROOT key ----------------------------------
        dflt = cfg.setdefault("defaults", {})
        if "OUTPUT_ROOT" in dflt and "output_root" not in dflt:
            dflt["output_root"] = dflt.pop("OUTPUT_ROOT")

        # …or grab from environment if absent
        dflt.setdefault("output_root", os.getenv("OUTPUT_ROOT", "outputs"))

        load_cfg._cache = cfg

    return load_cfg._cache


# ═════════════════════════ 2 · RUN-DIR RESOLVER ═════════════════
def resolve_run_dir(
    *,
    swan_year: str,
    run_tag  : Optional[str] = None,
    must_have: str | None    = None,
    create   : bool          = False,
) -> Path:
    """
    Locate (or create) the output *run* directory for a given event year.

    Folder layout
    -------------
        <output_root>/event=<YEAR>/<RUN_TAG>/

    If *run_tag* is **None** → choose the **latest** directory whose name
    matches ``r"run=\\d{3}"`` (run=001, run=002, …).

    If *must_have* is supplied, ensure that path (supports simple glob
    wild-cards) exists inside the resolved run-directory or raise
    *FileNotFoundError*.

    Returns
    -------
    **Path** pointing to the `<RUN_TAG>/` folder (never a stage sub-dir).
    """
    if swan_year in _OUTPUT_CACHE and run_tag is None and must_have is None:
        return _OUTPUT_CACHE[swan_year]

    cfg       = load_cfg()
    root_path = Path(cfg["defaults"]["output_root"]).expanduser().resolve()
    event_dir = root_path / f"event={swan_year}"

    if not event_dir.exists():
        if create:
            event_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"{event_dir} not found")

    # ―― choose / verify RUN_TAG ―――――――――――――――――――――――――――――――
    if run_tag is None:
        # newest run=NNN folder by mtime
        candidates = sorted(
            (p for p in event_dir.iterdir()
             if p.is_dir() and re.fullmatch(r"run=\d{3}", p.name)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No run directories under {event_dir}")
        run_dir = candidates[0]
        ulog.info("Auto-selected latest run directory: %s", run_dir.name)
    else:
        run_dir = event_dir / run_tag
        if not run_dir.exists():
            if create:
                run_dir.mkdir(parents=True, exist_ok=True)
                ulog.info("Created run directory %s", run_dir)
            else:
                raise FileNotFoundError(run_dir)

    # ―― must-have check (wild-cards allowed) ―――――――――――――――――――
    if must_have:
        # Convert e.g. ``stage03/*.csv`` to proper glob search
        pattern_path = run_dir / must_have
        hits = list(pattern_path.parent.glob(pattern_path.name))
        if not hits:
            raise FileNotFoundError(f"{must_have!s} not found in {run_dir}")

    _OUTPUT_CACHE[swan_year] = run_dir        # memoise
    return run_dir


# ═════════════════════════ 3 · SMALL HELPERS ════════════════════
def ensure_three_letter_tickers(df, id_col: str = "Symbol"):
    """
    Return *df* keeping **only** rows whose ticker is exactly three
    capital letters (discard secondary listings like ``BHP.AX``).
    """
    import pandas as pd

    if id_col not in df.columns:
        raise KeyError(f"{id_col} column missing in DataFrame")

    mask = df[id_col].astype(str).str.fullmatch(r"[A-Z]{3}")
    return df.loc[mask].copy()
