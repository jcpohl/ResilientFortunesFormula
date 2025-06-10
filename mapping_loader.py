"""
mapping_loader.py
─────────────────
Loads `ratio_domain_stage_map.yaml` (must live in the same folder)
and exposes `ratio_domain_stage_map` ready for import:

    from mapping_loader import ratio_domain_stage_map
"""
from pathlib import Path
from typing import Dict, List, Union   # ← add this for 3.8
import yaml

MAP_FILE = Path(__file__).with_name("ratio_domain_stage_map.yaml")

with MAP_FILE.open("r", encoding="utf-8") as fh:
    ratio_domain_stage_map: Dict[str, Union[str, List[str]]] = yaml.safe_load(fh)