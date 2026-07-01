from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent

def root_extract(morpheme_string: str):
    grammar_df = pd.read_csv('bantu_grammar_lookup.csv')

    known_affixes = set(grammar_df['morpheme_segment'].tolist())

    if pd.isna(morpheme_string):
        return []

    words = str(morpheme_string).split()
    roots = []

    for word in words:
        if '-' in word
            segments = word.replace('-', ' - ').split()

            for segment in segments:
                clean_seg = segment.replace('-', '')

                if clean_seg not in known_affixes and clean_seg != '':
                    roots.append(clean_seg)
        else:
            roots.append(word)
    return roots
