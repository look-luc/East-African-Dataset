from pathlib import Path

import pandas as pd

script_path = Path(__file__).resolve().parent.parent

languages = "ekegusii"
data_df = pd.read_csv(f"{script_path}/data/{languages}/data.csv", sep='\t')
if languages == "ganda":
    fig_rand_rows = pd.DataFrame(data_df[data_df["Output Type"]=="figurative_translate"]).sample(n=42)
    lit_rand_rows = pd.DataFrame(data_df[data_df["Output Type"]=="leteral_translate"]).sample(n=42)
else:
    fig_rand_rows = pd.DataFrame(data_df[data_df["Output Type"]=="figurative_translate"]).sample(n=40, replace=False)
    lit_rand_rows = pd.DataFrame(data_df[data_df["Output Type"]=="leteral_translate"]).sample(n=40, replace=False)
fig_rand_rows.to_csv(f"fig_{languages}_random_15.csv", sep='\t')
lit_rand_rows.to_csv(f"lit_{languages}_random_15.csv", sep='\t')
