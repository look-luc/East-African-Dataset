import pandas as pd
from datasets import load_dataset


def get_lang_data(lang:str):
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
    iso_code = bantu_iso_map.get(lang.lower())

    if not iso_code:
        print(f"Warning: {lang} is not in the target Bantu dictionary.")
        return pd.DataFrame()

    panlex_data = pd.DataFrame(load_dataset("cointegrated/panlex-meanings", name=iso_code, split="train"))
    eng_data = pd.DataFrame(load_dataset("cointegrated/panlex-meanings", name='eng', split="train"))

    panlex_df = panlex_data.merge(
        eng_data, on='meaning', suffixes=(f'_{iso_code}', '_eng')
    ).drop_duplicates(subset=[f'txt_{iso_code}', 'txt_eng'])

    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(r':', na=False)]

    noise_words = ['dollar', 'pound', 'shilling', 'republic', 'ocean', 'sea', 'continent', 'st.']
    noise_regex = '|'.join(noise_words)
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(noise_regex, case=False, na=False)]

    return panlex_df
