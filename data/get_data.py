import re
from pathlib import Path

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
kevin_Obote_few_shot = "kevin_Obote_few_shots"
zero_shot_experiment = "zero-shot experiment"

bantu_langs = ["ganda", "gikuyu", "tshiluba", "chiga", "tooro", "runyoro", "kamba"]

def making_df(jsonl_list):
    lang_pattern = r"\*\*Input\*\*:\nA proverb in (.*?)\n\n\*\*Output\*\*:"
    pattern = r"Now, please translate the following proverb:\n\n\*\*Input\*\*:\n(.*?)\n\n\*\*Output\*\*:"
    output_pattern = r"\*\*Output\*\*:<end_of_turn>\n<start_of_turn>model\n\", \"predict\": \"(.*?)\", \"label\"\: \"Avoid the fight which is not yours\.\""

    df = []
    for file_path in jsonl_list:
        temp_df = pd.read_json(file_path, lines=True)

        temp_df["source_file"] = file_path.name
        temp_df["experiment_config"] = file_path.parent.name

        df.append(temp_df)

    final_df = pd.concat(df, ignore_index=True)

    output_conditions = [
        final_df["experiment_config"].str.contains("literal", na=False),
        final_df["experiment_config"].str.contains("fig", na=False)
    ]

    output_choices = [
        "leteral_translate",
        "figurative_translate"
    ]

    final_df["Output Type"] = np.select(output_conditions, output_choices, default="unknown")

    final_df["language"] = final_df["prompt"].str.extract(lang_pattern, flags=re.DOTALL, expand=False)

    conditions = [
        final_df["language"].str.lower().isin(bantu_langs),
    ]
    choices = [
        "bantu",
    ]

    final_df["language_family"] = np.select(conditions, choices, default="Unknown")

    final_df["african_proverb"] = final_df["prompt"].str.extract(pattern, flags=re.DOTALL, expand=False)
    final_df["african_proverb"] = final_df["african_proverb"].str.strip()

    final_df["model output"] = final_df["output_pattern"].str.extract(pattern, flags=re.DOTALL, expand=False)

    return final_df[["experiment_config", "language", "language_family", "african_proverb", "label", "Output Type", "model output"]]


def Get_Data():
    kevin = script_dir / kevin_Obote_few_shot
    kevin_jsonl = list(kevin.rglob("*.jsonl"))
    kevin_df = pd.DataFrame(making_df(kevin_jsonl)).dropna()

    zero = script_dir / zero_shot_experiment
    zero_jsonl = list(zero.rglob("*.jsonl"))
    zero_df = pd.DataFrame(making_df(zero_jsonl)).dropna()

    data_df = pd.DataFrame(pd.concat([kevin_df, zero_df], ignore_index=True))

    data_df = data_df[data_df['language_family'] != 'Unknown']

    return data_df

if __name__ == "__main__":
    Get_Data()
