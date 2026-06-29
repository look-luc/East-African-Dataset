from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent
kevin_Obote_few_shot = "kevin_Obote_few_shots"
zero_shot_experiment = "zero-shot experiment"

def Get_Data():
    kevin = script_dir / kevin_Obote_few_shot
    kevin_jsonl = list(kevin.rglob("*.jsonl"))

    zero = script_dir / zero_shot_experiment
    zero_jsonl = list(zero.rglob("*.jsonl"))
