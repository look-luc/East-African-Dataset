import pandas as pd

import rough_morpheme.morpheme_counter as m_count
import rough_morpheme.morpheme_draft as md
from data.get_data import Get_Data


def main(path, task):
    if task == "get_data":
        print("Loading datasets...")
        data: pd.DataFrame = Get_Data()

        print("Generating rough morpheme breaks...")
        data["morpheme_breaks"] = md.segment(df=data, product_threshold=25)

        data.to_csv("data.csv", sep='\t')
        print("gathered data into data.csv")
    elif task =="count_morphemes":
        print("Counting Morphemes")

        morph_counter = m_count.morph_count(str(path / "data.csv"))
    elif task == "morpheme_translate":
        pass

    print("finished counting morphemes")
