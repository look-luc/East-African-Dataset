from pathlib import Path

import pandas as pd
import torch

import rough_morpheme.morpheme_counter as m_count
import rough_morpheme.morpheme_draft as md
from data.get_data import Get_Data
from morpheme_translate.final_pass import translation_final_pass
from morpheme_translate.model_segment import model_extract
from morpheme_translate.translate import translation
from semantic_score.Semantic_Score import compute_structural_metrics

script_dir = Path(__file__).resolve().parent.parent

def main(path, task:str, lang:str, proverbs):
    if task == "get_data":
        print("Loading datasets...")
        data = Get_Data()

        target_lang = str(lang).lower()

        print(f"Generating rough morpheme breaks for {target_lang}...")
        data["morpheme_breaks"] = md.segment(df=data, product_threshold=25)

        folder = Path(script_dir) / "data"
        folder.mkdir(parents=True, exist_ok=True)
        data.to_csv(folder / "data.csv", sep='\t')

        print("Gathered data into data.csv")
    elif task =="count_morphemes":
        print("Counting Morphemes")

        folder = Path(f"{script_dir}/data/{lang}")

        m_count.morph_count(str(folder / "data.csv"))

        print("finished counting morphemes")
    elif task=="morpheme_translate":
        folder = Path(f"{script_dir}/data/{lang}")
        folder.mkdir(parents=True, exist_ok=True)

        model_csv = folder / f"{lang.lower().capitalize()}_model_lem_seg.csv"
        if not model_csv.exists():
            print(f"Model file missing for {lang}. Running model_extract first...")
            model_extract(f"data/{lang}/data.csv", lang)
        translation(lang, lang, proverbs)
    elif task == "model_segment":
        model_extract(f"data/{lang}/data.csv", lang)
    elif task == "metrics":
        languages = ["ganda", "gikuyu", "ekegusii"]

        for l in languages:
            completed_path = script_dir / f"data/{l.lower()}/corrected/{l.lower()}/lit_{l.lower()}_random_15.csv"
            if not completed_path.exists():
                print(f"Completed file not found for {l} at {completed_path}. Run 'final_pass' first.")
                continue

            print(f"Computing metrics for {l} using {completed_path.name}...")
            try:
                df = pd.read_csv(completed_path)
            except Exception:
                df = pd.read_csv(completed_path, sep='\t')

            # Identify correction/manual column
            translation_column = ""
            for element in ["Correction", "corrected"]:
                if element in df.columns:
                    translation_column = element
                    break
            if translation_column == "":
                raise ValueError(f"Could not find a valid reference translation column in {completed_path}")

            # Ensure 'predict' column exists
            if 'predict' not in df.columns:
                for pred_candidate in ["label", "predict"]:
                    if pred_candidate in df.columns:
                        df['predict'] = df[pred_candidate]
                        break

            if 'predict' not in df.columns:
                raise KeyError(f"DataFrame for {l} is missing a prediction column ('predict' or 'translation').")

            compute_structural_metrics(l, df, translation_column)
    elif task == "final_pass":
        final_pass = translation_final_pass()
        if lang.lower() == "all":
            languages = ["ganda", "gikuyu", "chiga", "tooro", "runyoro", "kamba"]
            for langs in languages:
                print(f"----------------working on {langs}----------------")
                passes = final_pass.ranked_translation(fig_or_lit="lit", translation_keyword="translation", lang=langs)

                translation_df = pd.read_csv(f"{script_dir}/data/{langs.lower()}/lit_{langs.lower()}_random_15 - lit_{langs}_random_15.csv")
                translation_df["final pass"] = passes
                translation_df["final pass"] = translation_df["final pass"].fillna(translation_df["translation"])
                translation_df.to_csv(f"{script_dir}/data/{langs.lower()}/lit_{langs.lower()}_completed.csv", index=False)
        else:
            passes = final_pass.ranked_translation(fig_or_lit="lit", translation_keyword="translation", lang=lang)

            translation_df = pd.read_csv(f"{script_dir}/data/{lang.lower()}/lit_{lang.lower()}_random_15 - lit_{lang}_random_15.csv")
            translation_df["final pass"] = passes
            translation_df["final pass"] = translation_df["final pass"].fillna(translation_df["translation"])
            translation_df.to_csv(f"{script_dir}/data/{lang.lower()}/lit_{lang.lower()}_completed.csv", index=False)

        print(f"Saved complete dynamic translations for {lang}!")
