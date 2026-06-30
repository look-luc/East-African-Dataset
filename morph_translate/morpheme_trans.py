import os

import pandas as pd


def assemble_bantubvd_lexicon(path:str):
    cldf_dir = os.path.join(path, "cldf")

    if not os.path.exists(cldf_dir):
        raise FileNotFoundError(f"Could not find the cldf folder at {cldf_dir}. Did you run git clone?")

    print("Reading relational CLDF matrices...")

    df_forms = pd.read_csv(os.path.join(cldf_dir, "forms.csv"))
    df_parameters = pd.read_csv(os.path.join(cldf_dir, "parameters.csv"))
    df_languages = pd.read_csv(os.path.join(cldf_dir, "languages.csv"))

    df_concepts = df_parameters.rename(
        columns={
            'ID': 'Parameter_ID','Name': 'Concept'
        })[["Parameter_ID", "Concept"]]

    df_langs = df_languages.rename(columns={
            'ID': 'Language_ID',
            'ISO639P3code': 'Language_ISO',
            'Name': 'Language_Name'
        })[['Language_ID', 'Language_ISO', 'Glottocode', 'Language_Name']]

    print("Merging relational keys into a unified dictionary matrix...")

    lexicon_df = pd.merge(df_forms[['Language_ID', 'Parameter_ID', 'Form']], df_concepts, on='Parameter_ID', how='inner')
    lexicon_df = pd.merge(lexicon_df, df_langs, on='Language_ID', how='inner')

    lexicon_df['Form'] = lexicon_df['Form'].str.strip().str.lower()

    final_lexicon = lexicon_df[['Language_ISO', 'Glottocode', 'Form', 'Concept', 'Language_Name']].drop_duplicates()

    print(f"Compilation successful! Generated {len(final_lexicon)} clean lexical pairings.")
    return final_lexicon
