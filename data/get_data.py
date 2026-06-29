import re
from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent
kevin_Obote_few_shot = "kevin_Obote_few_shots"
zero_shot_experiment = "zero-shot experiment"

def making_df(jsonl_list):
    pattern = r"\*\*Input\*\*:\n(.*?)\n\n\*\*Output\*\*:"

    df = []
    for file_path in jsonl_list:
        temp_df = pd.read_json(file_path, lines=True)

        temp_df["source_file"] = file_path.name
        temp_df["experiment_config"] = file_path.parent.name

        df.append(temp_df)

    final_df = pd.concat(df, ignore_index=True)

    labels = final_df["label"]

    final_df["african_proverb"] = final_df["prompt"].str.extract(pattern, flags=re.DOTALL, expand=False)
    final_df["african_proverb"] = final_df["african_proverb"].str.strip()

    return labels, final_df


def Get_Data():
    kevin = script_dir / kevin_Obote_few_shot
    kevin_jsonl = list(kevin.rglob("*.jsonl"))
    kevin_labels, kevin_df = making_df(kevin_jsonl)

    zero = script_dir / zero_shot_experiment
    zero_jsonl = list(zero.rglob("*.jsonl"))
    zero_labels, zero_df = making_df(zero_jsonl)

if __name__ == "__main__":
    Get_Data()
