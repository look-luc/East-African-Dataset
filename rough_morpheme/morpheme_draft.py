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

def segment(df: DataFrame, threshold=12): # Raised threshold to minimize early cuts
    bantu_mask = df['language_family'] == 'bantu'
    global_trie = prep(df.loc[bantu_mask, "african_proverb"])

    morph = []
    for idx, row in df.iterrows():
        proverb = row["african_proverb"]
        lang_family = str(row["language_family"]).lower()

        if lang_family != 'bantu' or not isinstance(proverb, str) or proverb.strip() == "":
            morph.append("")
            continue

        words = re.findall(r'\b\w+\b', proverb.lower())
        segmented_words = []

        for word in words:
            segments = []
            current_segment = ""
            prefix = ""

            for i in range(len(word)):
                prefix += word[i]
                current_segment += word[i]

                next_choices_count = len(global_trie.get(prefix, set()))

                if next_choices_count >= threshold and i < len(word) - 1:
                    segments.append(current_segment)
                    current_segment = ""

            if current_segment:
                segments.append(current_segment)
            segmented_words.append("-".join(segments))

        morph.append(" ".join(segmented_words))

    return morph
