#!/usr/bin/env python
import os
import subprocess
import argparse
import yaml
import daily_setup  # Ensures RUN_DIR is set before running stages

def load_config(config_path="pipeline_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_stage_notebook(stage_file, env_vars):
    # Invoke papermill to run the notebook stage
    cmd = [
        "papermill",
        stage_file,
        stage_file.replace('.ipynb', '_output.ipynb')
    ]
    for k, v in env_vars.items():
        cmd.extend(["-p", k, str(v)])
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, help="Event year to run.")
    parser.add_argument("--run", required=True, help="Unique run identifier.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to config file.")
    args = parser.parse_args()

    # Load the pipeline configuration file
    config = load_config(args.config)

    # Derive environment variables from the configuration file
    INPUT_CSV   = config["defaults"]["INPUT_CSV"]
    OUTPUT_ROOT = config["defaults"]["OUTPUT_ROOT"]
    STAGE1_CFG  = config["defaults"].get("STAGE1_CFG", "")
    WIN_START   = config["events"][args.event]["WIN_START"]
    WIN_END     = config["events"][args.event]["WIN_END"]

    env_vars = {
        "INPUT_CSV": INPUT_CSV,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "STAGE1_CFG": STAGE1_CFG,
        "SWAN_YEAR": args.event,
        "WIN_START": WIN_START,
        "WIN_END": WIN_END,
        "RUN_TAG": args.run
    }

    # List of all completed stages
    stages = [
        "stage01.ipynb",
        "stage02.ipynb",
        "stage03.ipynb",
        "stage04.ipynb",
        "stage05a.ipynb",
        "stage05b.ipynb",        
        "stage06.ipynb",
        "stage07.ipynb",
        "stage08.ipynb",
        "stage09.ipynb",
        "stage10.ipynb",
        "stage11.ipynb",
        "stage12.ipynb",
        "stage13.ipynb",
        "stage14.ipynb",
        "stage15.ipynb",
        "stage16.ipynb"
    ]

    # Loop through each stage and run it
    for stage_file in stages:
        print(f"Running {stage_file} for event {args.event}...")
        run_stage_notebook(stage_file, env_vars)

if __name__ == "__main__":
    main()