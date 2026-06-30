import pandas as pd

import rough_morpheme.morpheme_counter as m_count
import rough_morpheme.morpheme_draft as md
from data.get_data import Get_Data
from morph_translate.morpheme_trans import assemble_bantubvd_lexicon


def main(path, task:str):
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
        ISO_MAP = {
            'ganda': 'lug', 'sukuma': 'suk', 'gikuyu': 'kik', 'pare': 'asu',
            'tshiluba': 'lua', 'zigula': 'zgu', 'kihangaza': 'hgz',
            'makonde': 'kde', 'nyaturu': 'rim', 'kihara': 'haa',
            'gweno': 'gwe', 'ekegusii': 'guz'
        }

        bantubvd_path = str(path.parent / "bantubvd")
        df_lexibank = assemble_bantubvd_lexicon(bantubvd_path)

        data = pd.read_csv(str(path / "data.csv"), sep='\t')

        bantu_df = data[data["language_family"].str.lower() == "bantu"].copy()

        language_series = pd.Series(bantu_df['language'])
        bantu_df['Language_ID'] = language_series.map(ISO_MAP.get)

        bantu_df = pd.DataFrame(bantu_df).dropna(subset=['Language_ID', 'morpheme_breaks'])

        bantu_df['Form'] = bantu_df['morpheme_breaks'].str.replace('-', ' ', regex=False).str.lower().str.split()
        bantu_exploded = bantu_df.explode('Form')

        df_joined = pd.merge(
            bantu_exploded,
            df_lexibank,
            left_on=['language_clean', 'Form'],
            right_on=['Language_Name_clean', 'Form'],
            how='inner'
        )

        print(f"Compilation complete. Matched rows found: {len(df_joined)}")
        if not df_joined.empty:
            print(df_joined[['language', 'morpheme_breaks', 'Form', 'Concept']].head())
        else:
            # Fallback diagnostics if alternative spelling forms exist (e.g., 'pare' vs 'asu')
            print("\n--- Diagnostic Name Alignment Check ---")
            print("Cleaned Names in data.csv:", sorted(bantu_exploded['language_clean'].unique()))
            print("Cleaned Names in Lexibank:", sorted(df_lexibank['Language_Name_clean'].unique()))

    print("finished counting morphemes")
