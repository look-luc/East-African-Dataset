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

def translation(file_name: str, lang: str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        print(f"Error: {lang} mapping not found.")
        return

    data_df = pd.read_csv(str(script_dir / "data.csv"), sep='\t')
    lang_data_df = pd.read_csv(str(script_dir / f"{file_name}.csv"), sep='\t')

    lang_txt_col = f"txt_{iso_code}"

    # Build a normalized list of dictionary pairs
    dict_entries = []
    for _, row in lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).iterrows():
        clean_key = str(row[lang_txt_col]).strip().lower().strip('-')
        dict_entries.append((clean_key, row['txt_eng']))

    # Convert to dictionary for O(1) exact lookups
    exact_translation_map = dict(dict_entries)

    lang_rows = data_df[data_df["language"].str.lower() == lang.lower()]

    output_data = {
        f'{lang.capitalize()} root': [],
        'English translation': []
    }

    roots = lang_rows['extracted_roots']
    for root in roots:
        root_list = []
        if isinstance(root, str):
            try:
                if root.startswith('[') and root.endswith(']'):
                    root_list = ast.literal_eval(root)
                else:
                    root_list = [root]
            except (ValueError, SyntaxError):
                root_list = [root]
        elif isinstance(root, list):
            root_list = root

        for root_item in root_list:
            clean_root = str(root_item).strip().lower().strip('-')
            if not clean_root:
                continue

            if clean_root in exact_translation_map:
                if clean_root == 'bal' and exact_translation_map[clean_root] == 'Mon':
                    pass
                else:
                    output_data[f'{lang.capitalize()} root'].append(root_item)
                    output_data['English translation'].append(exact_translation_map[clean_root])
                    continue

            if len(clean_root) > 3:
                match_found = False
                for dict_word, eng_trans in dict_entries:
                    if dict_word.startswith(clean_root) and len(dict_word) <= len(clean_root) + 3:
                        output_data[f'{lang.capitalize()} root'].append(root_item)
                        output_data['English translation'].append(eng_trans)
                        match_found = True
                        break
                if match_found:
                    continue

    df_out = pd.DataFrame(output_data).drop_duplicates().reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Saved hybrid translations to {lang.lower()}_translated.csv with {len(df_out)} unique entries.")
