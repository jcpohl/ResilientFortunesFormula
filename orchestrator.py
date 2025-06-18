#!/usr/bin/env python
"""
Event-aware notebook orchestrator for the Resilient-Fortunes pipeline
v5 · 2025-06-20
────────────────────────────────────────────────────────────────────
• Adds SAVE_FORMAT / PURGE_OLD support.
• Forwards SAVE_FORMAT to every stage via env.
Examples
--------
python orchestrator.py --event 2008         # single event
python orchestrator.py --event ALL          # all events
"""
from __future__ import annotations

import argparse, datetime as _dt, os, shutil, subprocess, sys, yaml, nbformat
from pathlib  import Path
from typing   import Dict


# ───────────────────────── helpers ──────────────────────────
def load_config(path: str = "pipeline_config.yaml") -> Dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _ensure_kernelspec(nb_path: str, kernel: str = "python3") -> None:
    """Add a kernelspec if the notebook metadata is missing."""
    nb = nbformat.read(nb_path, as_version=4)
    if "kernelspec" not in nb.metadata:
        nb.metadata["kernelspec"] = {
            "name": kernel,
            "display_name": "Python 3",
            "language": "python",
        }
        nbformat.write(nb, nb_path)


def run_notebook(nb: str, env: Dict[str, str], run_dir: Path) -> None:
    """
    Execute *nb* via Papermill and save the executed copy under:
        <run_dir>/<stage-folder>/<nbstem>_<SWAN>_<TAG>_output.ipynb
    """
    _ensure_kernelspec(nb)                       # safeguard
    os.environ.update(env)

    stage_dir = run_dir / Path(nb).stem.lower()  # e.g. stage04c
    stage_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{env['SWAN_YEAR']}_{env['RUN_TAG']}_output.ipynb"
    out_nb = stage_dir / f"{Path(nb).stem}_{suffix}"

    cmd = ["papermill", nb, str(out_nb)]
    # forward params (avoid papermill bug with dashes in names)
    for k, v in env.items():
        if k != "RUN_TAG":
            cmd += ["-p", k, str(v)]

    print("▶︎", *cmd, flush=True)
    subprocess.run(cmd, check=True)


# ───────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True,
                    help="SWAN year to run, or ALL for every event")
    ap.add_argument("--run", default=_dt.date.today().isoformat(),
                    help="Run-tag / folder name (default = today)")
    ap.add_argument("--config", default="pipeline_config.yaml",
                    help="Path to YAML config")
    args = ap.parse_args()

    cfg         = load_config(args.config)
    events_cfg  = cfg["events"]
    dflt        = cfg.get("defaults", {})

    # ★ new - global options
    SAVE_FORMAT = dflt.get("SAVE_FORMAT", "csv").lower()      # 'csv' | 'parquet'
    PURGE_OLD   = bool(dflt.get("PURGE_OLD", False))

    # decide which events to run
    if args.event.upper() == "ALL":
        swan_years = list(events_cfg.keys())
    else:
        if args.event not in events_cfg:
            sys.exit(f"[ERROR] event {args.event} not in pipeline_config.yaml")
        swan_years = [args.event]

    # execution roster (add earlier stages here if needed)
    STAGES = ["stage01.ipynb",
              "stage02.ipynb",
                "stage03.ipynb",
                "stage04.ipynb",
                "stage04B.ipynb",
                "stage04C.ipynb",
                "stage04d.ipynb",
                "stage05a.ipynb",
                "stage05b.ipynb",
                "stage05c.ipynb",
                "stage06.ipynb",
                "stage06b.ipynb",
                "stage07.ipynb",
    ]

    INPUT_CSV   = dflt["INPUT_CSV"]
    OUTPUT_ROOT = dflt["OUTPUT_ROOT"]
    STAGE1_CFG  = dflt.get("STAGE1_CFG", "")

    for swan in swan_years:
        print(f"\n================  RUNNING EVENT {swan}  =================",
              flush=True)
        win = events_cfg[swan]

        env: Dict[str, str] = {
            "INPUT_CSV":   INPUT_CSV,
            "OUTPUT_ROOT": OUTPUT_ROOT,
            "SWAN_YEAR":   swan,
            "WIN_START":   str(win["WIN_START"]),
            "WIN_END":     str(win["WIN_END"]),
            "RUN_TAG":     args.run,
            "SAVE_FORMAT": SAVE_FORMAT,        # ★ forward to stages
        }
        if STAGE1_CFG:
            env["STAGE1_CFG"] = STAGE1_CFG

        run_dir = Path(OUTPUT_ROOT) / f"event={swan}" / args.run
        run_dir.mkdir(parents=True, exist_ok=True)

        # ― execute notebooks ―
        for nb in STAGES:
            print(f"\n─── executing {nb} for SWAN {swan} (run {args.run}) ───",
                  flush=True)
            run_notebook(nb, env, run_dir)

        # ― optional purge of older runs ―
        if PURGE_OLD:
            event_dir = run_dir.parent
            for older in event_dir.iterdir():
                if older.is_dir() and older.name != args.run:
                    print(f"🗑  PURGE_OLD: deleting {older}", flush=True)
                    shutil.rmtree(older, ignore_errors=True)

        print(f"\n✓ Event {swan} finished → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
