from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from morpheme_translate.extraction import root_extract

script_dir = Path(__file__).resolve().parent.parent

bantu_iso_map = {
    'nyaturu': 'rim', 'bangubangu': 'bby', 'kwele': 'kwl', 'kihangaza': 'han',
    'soga': 'xog', 'pare': 'asa', 'olusamia': 'lsm', 'taabwa': 'tap',
    'nande': 'nnb', 'hemba': 'hem', 'tshiluba': 'lua', 'tooro': 'ttj',
    'hema': 'hea', 'holoholo': 'hoo', 'meru': 'mer', 'runyoro': 'nyo',
    'zigula': 'ziw', 'makonde': 'kde', 'kamba': 'kam', 'digo': 'dig',
    'kihara': 'haq', 'nyala': 'nle', 'gikuyu': 'kik', 'tetela': 'tll',
    'rufumbira': 'kin', 'sukuma': 'suk', 'ganda': 'lug', 'gweno': 'gwe',
    'chiga': 'cgg', 'ekegusii': 'guz'
}

@lru_cache(maxsize=64)
def get_lang_data(lang:str):
    iso_code = bantu_iso_map.get(lang.lower())

    if not iso_code:
        print(f"Warning: {lang} is not in the target Bantu dictionary.")
        return pd.DataFrame()

    panlex_data = load_dataset("cointegrated/panlex-meanings", name=iso_code, split="train") \
        .select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()

    eng_data = load_dataset("cointegrated/panlex-meanings", name='eng', split="train") \
        .select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()

    panlex_df = panlex_data.merge(
        eng_data, on='meaning', how='left', suffixes=(f'_{iso_code}', '_eng')
    ).drop_duplicates(subset=[f'txt_{iso_code}', 'txt_eng'])

    noise = ['dollar', 'pound', 'shilling', 'republic', 'ocean', 'sea', 'continent', 'st.', 'saudi', 'papua', 'zimbabwe', 'sudanese']
    noise_regex = '|'.join(noise)
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(noise_regex, case=False, na=False)]

    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(r':|/', na=False)]

    panlex_df = panlex_df[panlex_df['txt_eng'].str.len() < 100]
    panlex_df = panlex_df[panlex_df['langvar_uid_eng'] == 'eng-000']
    return panlex_df

def translation(file_name:str, lang:str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        print(f"Error: {lang} mapping not found.")
        return

    data_df = pd.read_csv(str(script_dir / "data.csv"), sep='\t')
    lang_data_df = pd.read_csv(str(script_dir / f"{file_name}.csv"), sep='\t')

    bantu_grammar_df = pd.read_csv(str(script_dir / "bantu_grammar_lookup.csv"))
    bantu_grammar_lang_df = bantu_grammar_df[bantu_grammar_df['language'] == lang].copy()

    lang_txt_col = f"txt_{iso_code}"

    translation_map = dict(zip(lang_data_df[lang_txt_col], lang_data_df['txt_eng']))

    lang_rows = data_df[data_df["language"] == lang]

    if "proposed_leipzig_gloss" not in bantu_grammar_lang_df.columns:
        bantu_grammar_lang_df["proposed_leipzig_gloss"] = None

    for idx, row in lang_rows.iterrows():
        roots = row['extracted_roots']
        for root in roots:
            if root in translation_map:
                if idx in bantu_grammar_lang_df.index:
                    bantu_grammar_lang_df.at[idx, "proposed_leipzig_gloss"] = translation_map[root]

    output_path = script_dir / f"{file_name}_translated.csv"
    bantu_grammar_lang_df.to_csv(output_path, index=False)
    print(f"Saved translations to {output_path}")
