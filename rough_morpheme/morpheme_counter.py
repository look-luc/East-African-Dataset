import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

def parse_morphemes(word):
    parts = re.split(r'(-)', word)

    if len(parts) == 1:
        return [word]

    morphemes = []
    for i in range(len(parts)):
        if parts[i] == '-':
            continue

        if i + 1 < len(parts) and parts[i+1] == '-':
            morphemes.append(parts[i] + '-')

        if i - 1 >= 0 and parts[i-1] == '-':
            morphemes.append('-' + parts[i])

    return morphemes

def morph_count(df_path:str, output_file:str="bantu_grammar_lookup.csv", top_n_threshold=1000000):
    df = pd.read_csv(df_path, sep='\t')

    print(f"Reading dataset: {df_path}...")
    bantu_df = df[df['language_family'].str.lower() == 'bantu'].copy()
    langs = np.unique(bantu_df["language"]).tolist()

    grammatical_record = []

    print(f"Processing morpheme distributions across {len(langs)} Bantu languages...")

    for lang in langs:
        lang_df = pd.DataFrame(bantu_df[bantu_df['language'] == lang])
        all_morpheme = []

        for string in lang_df['morpheme_breaks'].dropna():
            for word in str(string).split():
                morphemes = parse_morphemes(word)
                all_morpheme.extend(morphemes)

        morpheme_count:Counter = Counter(all_morpheme)
        total_tokens_counted:int = len(all_morpheme)

        most_common_morpheme = morpheme_count.most_common(top_n_threshold)

        for rank, (morpheme, count) in enumerate(most_common_morpheme, start=1):
            rel_freq = (count / total_tokens_counted) * 100

            grammatical_record.append({
                "language": lang,
                "morpheme_segment": morpheme,
                "raw_count": count,
                "relative_frequency_pct": round(rel_freq, 3),
                "frequency_rank": rank,
                "proposed_leipzig_gloss": ""
            })
    look_up = pd.DataFrame(grammatical_record)

    look_up.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Success! High-frequency grammar matrix exported to: {output_file}")
    print(f"Total reference rows generated: {len(look_up)}")

    return look_up
