import re
from collections import defaultdict

import pandas as pd
from pandas import DataFrame


def prep(text:pd.Series):
    trie = defaultdict(set)

    all_text = " ".join(text.dropna().astype(str)).lower()
    words = re.findall(r'\b\w+\b', all_text)

    for word in words:
        for i in range(1, len(word)):
            prefix = word[:i]
            next_char = word[i]
            trie[prefix].add(next_char)

    return trie

def segment(df:DataFrame, threshold=3):
    bantu_mask = df['language_family'] == 'bantu'
    global_trie = prep(df.loc[bantu_mask, "african_proverb"])

    morph = []
    for idx, row in df.iterrows():
        lang_family_str = str(row["language_family"]).lower()
        proverb_val = row["african_proverb"]

        # If it's not a Bantu language, or if the text is empty, skip it safely
        if lang_family_str != 'bantu' or not isinstance(proverb_val, str) or proverb_val.strip() == "":
            morph.append("")
            continue

        words = re.findall(r'\b\w+\b', str(proverb_val).lower())
        segmented_words = []

        for word in words:
            segments = []
            current_chunk = ""

            for i in range(len(word)):
                current_chunk += word[i]

                next_choices_count = len(global_trie.get(current_chunk, set()))

                if next_choices_count >= threshold and i < len(word) - 1:
                    segments.append(current_chunk)
                    current_chunk = ""

            if current_chunk:
                segments.append(current_chunk)
            segmented_words.append("-".join(segments))

        morph.append(" ".join(segmented_words))

    return morph
