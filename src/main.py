from pathlib import Path

import rough_morpheme.morpheme_counter as m_count
import rough_morpheme.morpheme_draft as md
from data.get_data import Get_Data
from morpheme_translate.model_segment import model_extract
from morpheme_translate.translate import translation

script_dir = Path(__file__).resolve().parent.parent

def main(path, task:str, lang:str):
    if task == "get_data":
        print("Loading datasets...")
        data = Get_Data()

        # Determine if the target language uses augments
        target_lang = str(lang).lower()

        print(f"Generating rough morpheme breaks for {target_lang}...")
        # Pass down the configuration flag to your segmentation tool
        data["morpheme_breaks"] = md.segment(df=data, product_threshold=25)

        folder = Path(script_dir) / "data"
        folder.mkdir(parents=True, exist_ok=True)
        data.to_csv(folder / "data.csv", sep='\t')

        print("Gathered data into data.csv")
    elif task =="count_morphemes":
        print("Counting Morphemes")

        folder = Path(f"{script_dir}/data")

        m_count.morph_count(str(folder / "data.csv"))

        print("finished counting morphemes")
    elif task=="morpheme_translate":
        folder = Path(f"{script_dir}/data/{lang}")
        folder.mkdir(parents=True, exist_ok=True)

        model_csv = str(folder / f"{lang.lower().capitalize()}_model_lem_seg.csv")
        if not model_csv.exists():
            print(f"Model file missing for {lang}. Running model_extract first...")
            model_extract("data/data.csv", lang)
        translation(lang, lang)
    elif task == "model_segment":
        model_extract("data/data.csv", lang)
