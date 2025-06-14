#!/usr/bin/env python
"""
Event-aware orchestrator for the Resilient-Fortunes pipeline.
Runs Stage-01 … Stage-03 and Stage-25 for one or several SWAN events.

Usage
-----
# run a single event
python orchestrator.py --event 2008 --run 2025-06-13

# run every event listed in pipeline_config.yaml
python orchestrator.py --event ALL --run 2025-06-13
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os, sys, subprocess, yaml
from pathlib import Path
from typing import Dict

# ───────────────────────── helpers ──────────────────────────
def load_config(path: str = "pipeline_config.yaml") -> Dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def run_notebook(nb: str, env_vars: Dict[str, str]) -> None:
    """
    Execute *nb* with Papermill.
    • All env_vars are propagated to the subprocess environment.
    • Every var is also passed as `-p name value`, EXCEPT RUN_TAG
      (we do not inject it to avoid the double-dash bug in folder names).
    """
    os.environ.update(env_vars)

    cmd = ["papermill", nb, f"{Path(nb).stem}_output.ipynb"]
    for k, v in env_vars.items():
        if k == "RUN_TAG":
            continue            # avoid “2025--06-13” duplication
        cmd.extend(["-p", k, str(v)])

    print("▶︎", *cmd)
    subprocess.run(cmd, check=True)

# ───────────────────────── main ─────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True,
                        help="SWAN year to run, or ALL to loop over every "
                             "event in pipeline_config.yaml")
    parser.add_argument("--run", default=_dt.date.today().isoformat(),
                        help="Run-tag / folder name (default = today YYYY-MM-DD)")
    parser.add_argument("--config", default="pipeline_config.yaml",
                        help="Path to YAML config")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    events_cfg  = cfg["events"]

    if args.event == "ALL":
        swan_years = list(events_cfg.keys())
    else:
        if args.event not in events_cfg:
            print(f"[ERROR] event {args.event} not found in pipeline_config.yaml",
                  file=sys.stderr)
            sys.exit(1)
        swan_years = [args.event]

    stages = ["stage01.ipynb", "stage02.ipynb", "stage03.ipynb", "stage25.ipynb"]

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

        # create run folder early so stage-01 has somewhere to write logs
        run_dir = Path(OUTPUT_ROOT) / f"event={swan}" / args.run
        run_dir.mkdir(parents=True, exist_ok=True)

        for nb in stages:
            print(f"\n─── Executing {nb} for event {swan} (run {args.run}) ───")
            run_notebook(nb, env_vars)

        print(f"✓ Event {swan} finished → {run_dir}")

if __name__ == "__main__":
    main()