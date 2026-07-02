import ast
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset

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

def normalize_ortho(lemma: str):
    lemma = lemma.strip().lower().strip('-')

    if lemma.startswith('um'):
        lemma = 'om' + lemma[2:]
    elif lemma.startswith('un'):
        lemma = 'on' + lemma[2:]
    elif lemma.startswith('in'):
        lemma = 'en' + lemma[2:]
    elif lemma.startswith('im'):
        lemma = 'em' + lemma[2:]

    return lemma

def translation(file_name: str, lang: str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        print(f"Error: {lang} mapping not found.")
        return

    data_df = pd.read_csv(str(script_dir / "data.csv"), sep='\t')
    lang_data_df = pd.read_csv(str(script_dir / f"{file_name}.csv"), sep='\t')

    lang_txt_col = f"txt_{iso_code}"

    model_path = script_dir / f"{file_name}.csv"
    if not model_path.exists():
        model_path = script_dir / f"{lang.lower()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    dict_entries = {}
    for _, row in lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).iterrows():
        clean_dict_key = str(row[lang_txt_col]).strip().lower().strip('-')
        dict_entries[clean_dict_key] = row['txt_eng']

    exact_translation_map = dict(dict_entries)

    lang_rows = data_df[data_df["language"].str.lower() == lang.lower()]

    if model_df.columns[0] == 'Unnamed: 0':
        model_df.rename(columns={'Unnamed: 0': 'surface_word'}, inplace=True)
    else:
        model_df.index.name = 'surface_word'
        model_df = model_df.reset_index()

    output_data = {
        f'{lang.capitalize()} root': [],
        f'{lang.capitalize()} Lemma': [],
        'English translation': []
    }

    for _, row in model_df.iterrows():
        surface_word = str(row['surface_word'])
        raw_lemmas = row['lemmatization']

        # Safely parse the literal string representation of the Python list
        try:
            lemma_list = ast.literal_eval(raw_lemmas)
            if not isinstance(lemma_list, list) or not lemma_list:
                continue
            model_lemma = str(lemma_list[0])
        except (ValueError, SyntaxError, IndexError):
            continue

        normalized_lemma = normalize_ortho(model_lemma)

        if normalized_lemma in exact_translation_map:
            output_data['Surface Word'].append(surface_word)
            output_data[f'{lang.capitalize()} Lemma'].append(normalized_lemma)
            output_data['English translation'].append(exact_translation_map[normalized_lemma])

        elif len(normalized_lemma) > 3:
            matched_fallback = False
            for dict_word, eng_trans in exact_translation_map.items():
                if dict_word.startswith(normalized_lemma) and len(dict_word) <= len(normalized_lemma) + 3:
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                    output_data['English translation'].append(eng_trans)
                    matched_fallback = True
                    break
            if matched_fallback:
                continue

    df_out = pd.DataFrame(output_data).drop_duplicates().reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Successfully processed model data. Saved entries to {lang.lower()}_translated.csv with {len(df_out)} unique pairs.")
