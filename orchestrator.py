#!/usr/bin/env python
"""
Event-aware orchestrator for the Resilient-Fortunes pipeline
────────────────────────────────────────────────────────────
Examples
--------
# single event
python orchestrator.py --event 2008 

# every event in pipeline_config.yaml
python orchestrator.py --event ALL  
"""

## python orchestrator.py --event ALL  ** use this prompt to run - uses todays date as run tag

from __future__ import annotations

import argparse, datetime as _dt, os, subprocess, sys, yaml
from pathlib import Path
from typing import Dict

# ───────────────────────── helpers ──────────────────────────
def load_config(path: str = "pipeline_config.yaml") -> Dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_notebook(nb: str, env_vars: Dict[str, str]) -> None:
    """
    Execute *nb* with Papermill and write a **unique** output .ipynb
    (adds SWAN_YEAR and RUN_TAG to the file-name so Windows never locks it).

    • All env_vars are forwarded to the subprocess environment.
    • Each var is also passed as `-p name value`, **except** RUN_TAG
      (avoids double-dash bug in papermill).
    """
    os.environ.update(env_vars)

    swan = env_vars.get("SWAN_YEAR", "NA")
    run  = env_vars.get("RUN_TAG", "")
    out_ipynb = f"{Path(nb).stem}_{swan}_{run}_output.ipynb"

    cmd = ["papermill", nb, out_ipynb]
    for k, v in env_vars.items():
        if k == "RUN_TAG":
            continue                     # skip RUN_TAG as papermill parameter
        cmd.extend(["-p", k, str(v)])

    print("▶︎", *cmd)
    subprocess.run(cmd, check=True)


# ───────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True,
                    help="SWAN year to run, or ALL for every event in the config")
    ap.add_argument("--run", default=_dt.date.today().isoformat(),
                    help="Run-tag / folder name (default = today YYYY-MM-DD)")
    ap.add_argument("--config", default="pipeline_config.yaml",
                    help="Path to YAML config")
    args = ap.parse_args()

    cfg        = load_config(args.config)
    events_cfg = cfg["events"]

    # Which events?
    if args.event.upper() == "ALL":
        swan_years = list(events_cfg.keys())
    else:
        if args.event not in events_cfg:
            sys.exit(f"[ERROR] event {args.event} not in pipeline_config.yaml")
        swan_years = [args.event]

    stages = ["stage01.ipynb",
                "stage02.ipynb",
                "stage03.ipynb",
                "stage04.ipynb",
                "stage04B.ipynb",
                "stage05a.ipynb",
                "stage05b.ipynb",
                "stage06.ipynb",
                "stage07.ipynb",
                "stage08.ipynb",
                "stage09.ipynb",
                "stage10.ipynb",
                "stage11.ipynb",
                "stage14.ipynb",
                "stage17.ipynb",  
                "stage25.ipynb"]

    INPUT_CSV   = cfg["defaults"]["INPUT_CSV"]
    OUTPUT_ROOT = cfg["defaults"]["OUTPUT_ROOT"]
    STAGE1_CFG  = cfg["defaults"].get("STAGE1_CFG", "")

    for swan in swan_years:
        print(f"\n================  RUNNING EVENT {swan}  =================")
        win = events_cfg[swan]

        env_vars: Dict[str, str] = {
            "INPUT_CSV":   INPUT_CSV,
            "OUTPUT_ROOT": OUTPUT_ROOT,
            "SWAN_YEAR":   swan,
            "WIN_START":   str(win["WIN_START"]),
            "WIN_END":     str(win["WIN_END"]),
            "RUN_TAG":     args.run,
        }
        if STAGE1_CFG:
            env_vars["STAGE1_CFG"] = STAGE1_CFG

        # Pre-create the run folder so Stage-01 has somewhere to log
        run_dir = Path(OUTPUT_ROOT) / f"event={swan}" / args.run
        run_dir.mkdir(parents=True, exist_ok=True)

        for nb in stages:
            print(f"\n─── Executing {nb} for event {swan} (run {args.run}) ───")
            run_notebook(nb, env_vars)

        print(f"✓ Event {swan} finished → {run_dir}")

if __name__ == "__main__":
    main()