from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent.parent

def root_extract(morpheme_string: str, language: str):
    grammar_df = pd.read_csv(f'{script_dir}/data/bantu_grammar_lookup.csv')

    lang_grammar = grammar_df[grammar_df['language'] == str(language).lower()]
    known_affixes = set(lang_grammar['morpheme_segment'].tolist())

    if pd.isna(morpheme_string):
        return []

    words = str(morpheme_string).split()
    roots = []

    for word in words:
        if '-' in word:
            segments = word.replace('-', ' - ').split()
            for segment in segments:
                clean_seg = segment.replace('-', '').strip()

                if clean_seg not in known_affixes and clean_seg != '':
                    roots.append(clean_seg)
        else:
            roots.append(word)
    return roots
