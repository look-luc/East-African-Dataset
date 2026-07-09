from pathlib import Path

import pandas as pd

script_path = Path(__file__).resolve().parent.parent

languages = ["ganda", "gikuyu", "tshiluba", "chiga", "tooro", "runyoro", "kamba"]
data_df = pd.read_csv(f"{script_path}/data/data.csv", sep='\t')

for lang in languages:
    lang_df = data_df[data_df["language"]==lang]
    if lang == "ganda":
        fig_rand_rows = pd.DataFrame(lang_df[lang_df["Output Type"]=="figurative_translate"]).sample(n=42)
        lit_rand_rows = pd.DataFrame(lang_df[lang_df["Output Type"]=="leteral_translate"]).sample(n=42)
    else:
        fig_rand_rows = pd.DataFrame(lang_df[lang_df["Output Type"]=="figurative_translate"]).sample(n=40)
        lit_rand_rows = pd.DataFrame(lang_df[lang_df["Output Type"]=="leteral_translate"]).sample(n=40)
    fig_rand_rows.to_csv(f"fig_{lang}_random_15.csv", sep='\t')
    lit_rand_rows.to_csv(f"lit_{lang}_random_15.csv", sep='\t')
