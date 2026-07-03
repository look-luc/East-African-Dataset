import rough_morpheme.morpheme_counter as m_count
import rough_morpheme.morpheme_draft as md
from data.get_data import Get_Data
from morpheme_translate.model_segment import model_extract
from morpheme_translate.translate import get_lang_data, translation


def main(path, task:str):
    if task == "get_data":
        print("Loading datasets...")
        data, full_data = Get_Data()

        print("Generating rough morpheme breaks...")
        data["morpheme_breaks"] = md.segment(df=data, product_threshold=25)

        data.to_csv("data.csv", sep='\t')
        full_data.to_csv("full_data.csv", sep='\t')
        print("gathered data into data.csv")
    # elif task =="count_morphemes":
    #     print("Counting Morphemes")
    #     morph_counter = m_count.morph_count(str(path / "data.csv"))

    #     print("finished counting morphemes")
    elif task=="morpheme_translate":
        translation("ganda", "ganda")
    elif task == "model_segment":
        model_extract("data.csv", "Ganda")
