import re
from pathlib import Path

import numpy as np
import pandas as pd

script_dir = Path(__file__).resolve().parent
kevin_Obote_few_shot = "kevin_Obote_few_shots"
zero_shot_experiment = "zero-shot experiment"

bantu_langs = [
    "bangubangu",
    "chiga",
    "digo",
    "ekegusii",
    "ganda",
    "gikuyu",
    "gweno",
    "hema",
    "hemba",
    "holoholo",
    "kamba",
    "kihangaza",
    "kihara",
    "kwele",
    "makonde",
    "meru",
    "nande",
    "nyala",
    "nyaturu",
    "olusamia",
    "pare",
    "rufumbira",
    "runyoro",
    "soga",
    "sukuma",
    "taabwa",
    "tetela",
    "tooro",
    "tshiluba",
    "zigula",
]

nilotic_langs = [
    "alur",
    "luo",
    "maasai",
    "samburu",
    "teso",
    "turkana",
    "nandi",
    "tugen",
]

nubian_lang = [
    "nubian",
    "nubian_2",
]

cushitic_lang = [
    "somali",
    "borana",
    "burji",
    "orma",
    "rendille",
]

def making_df(jsonl_list, output_type:str):
    lang_pattern = r"\*\*Input\*\*:\nA proverb in (.*?)\n\n\*\*Output\*\*:"
    pattern = r"Now, please translate the following proverb:\n\n\*\*Input\*\*:\n(.*?)\n\n\*\*Output\*\*:"

    df = []
    for file_path in jsonl_list:
        temp_df = pd.read_json(file_path, lines=True)

        temp_df["source_file"] = file_path.name
        temp_df["experiment_config"] = file_path.parent.name

        df.append(temp_df)

    temp = pd.concat(df, ignore_index=True)
    final_df = pd.concat(df, ignore_index=True)

    final_df["language"] = final_df["prompt"].str.extract(lang_pattern, flags=re.DOTALL, expand=False)
    final_df["Output Type"] = output_type

    conditions = [
        final_df["language"].str.lower().isin(bantu_langs),
        final_df["language"].str.lower().isin(nilotic_langs),
        final_df["language"].str.lower().isin(cushitic_lang),
        final_df["language"].str.lower().isin(nubian_lang)
    ]
    choices = [
        "bantu",
        "nilotic",
        "nubian",
        "cushitic",
    ]

    final_df["language_family"] = np.select(conditions, choices, default="Unknown")

    final_df["african_proverb"] = final_df["prompt"].str.extract(pattern, flags=re.DOTALL, expand=False)
    final_df["african_proverb"] = final_df["african_proverb"].str.strip()

    return final_df[["language", "language_family", "african_proverb", "label"]],temp


def Get_Data():
    kevin = script_dir / kevin_Obote_few_shot
    kevin_jsonl = list(kevin.rglob("*.jsonl"))
    kevin_df, temp = making_df(kevin_jsonl, "leteral_translate")

    zero = script_dir / zero_shot_experiment
    zero_jsonl = list(zero.rglob("*.jsonl"))
    zero_df, temp_zero = making_df(zero_jsonl, "figurative_translate")

    full_data = pd.DataFrame(pd.concat([temp, temp_zero], ignore_index=True))
    data_df = pd.DataFrame(pd.concat([kevin_df, zero_df], ignore_index=True))

    return data_df, full_data

if __name__ == "__main__":
    Get_Data()
