import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import rough_morpheme.morpheme_draft as md
from data.get_data import Get_Data


def main():
    print("Loading datasets...")
    data: pd.DataFrame = Get_Data()

    print("Generating rough morpheme breaks...")
    data["morpheme_breaks"] = md.segment(df=data, strategy="product")

    print(data[["language", "african_proverb", "morpheme_breaks"]].head(10))

    data.to_csv("data.csv", sep='\t')
    print("gathered data into data.csv")
