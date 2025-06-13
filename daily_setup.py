#!/usr/bin/env python
import os
from pathlib import Path
from datetime import datetime

# Define the base output directory (can be overridden via environment variable)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "outputs_rff")).expanduser()

# Only create a new run folder if RUN_DIR is not already set
if os.getenv("RUN_DIR") is None:
    # Create a new folder inside OUTPUT_ROOT/daily with today's UTC date (YYYY-MM-DD)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    new_run_dir = OUTPUT_ROOT / "daily" / today
    new_run_dir.mkdir(parents=True, exist_ok=True)
    # Set the RUN_DIR environment variable for all subsequent stages
    os.environ["RUN_DIR"] = str(new_run_dir)
    print(f"New RUN_DIR for today created: {new_run_dir}")
else:
    print(f"Using existing RUN_DIR: {os.environ['RUN_DIR']}")