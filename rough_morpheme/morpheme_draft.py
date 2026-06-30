import re
from collections import defaultdict

import pandas as pd
from pandas import DataFrame


def prep_dual(text: pd.Series):
    """
    Builds both a forward trie (for tracking successor variety from left-to-right)
    and a backward trie (for tracking predecessor variety from right-to-left).
    """
    forward_trie = defaultdict(set)
    backward_trie = defaultdict(set)

    all_text = " ".join(text.dropna().astype(str)).lower()
    words = re.findall(r'\b\w+\b', all_text)

    for word in words:
        if len(word) < 2:
            continue

        # 1. Populate Forward Trie (Prefixes)
        for i in range(1, len(word)):
            prefix = word[:i]
            next_char = word[i]
            forward_trie[prefix].add(next_char)

        # 2. Populate Backward Trie (Reversed suffixes)
        rev_word = word[::-1]
        for i in range(1, len(rev_word)):
            rev_prefix = rev_word[:i]
            prev_char = rev_word[i]
            backward_trie[rev_prefix].add(prev_char)

    return forward_trie, backward_trie

def segment(df: DataFrame, strategy="product", min_word_len=4):
    """
    Segments words using the intersection of forward and backward variety scores.

    Strategies available:
      - "product"     : Multiplies forward and backward counts, then finds peaks.
      - "union"       : Finds peaks in both directions independently and takes their union.
      - "intersection": Finds peaks in both directions independently and takes their intersection.
    """
    bantu_mask = df['language_family'] == 'bantu'
    forward_trie, backward_trie = prep_dual(df.loc[bantu_mask, "african_proverb"])

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
            if len(word) < min_word_len:
                segmented_words.append(word)
                continue

            num_boundaries = len(word) - 1
            FORWARD_COUNT = []
            BACKWARD_COUNT = []

            # Calculate variety scores at every internal character boundary
            for k in range(num_boundaries):
                prefix = word[:k+1]
                suffix_rev = word[k+1:][::-1]

                FORWARD_COUNT.append(len(forward_trie.get(prefix, set())))
                BACKWARD_COUNT.append(len(backward_trie.get(suffix_rev, set())))

            cut_positions = set()

            if strategy == "product":
                scores = [FORWARD_COUNT[k] * BACKWARD_COUNT[k] for k in range(num_boundaries)]
                for k in range(num_boundaries):
                    left_val = scores[k-1] if k > 0 else -1
                    right_val = scores[k+1] if k < num_boundaries - 1 else -1
                    if scores[k] > left_val and scores[k] > right_val and scores[k] > 1:
                        cut_positions.add(k)

            elif strategy in ("union", "intersection"):
                forward_peaks = set()
                backward_peaks = set()

                for k in range(num_boundaries):
                    f_left = FORWARD_COUNT[k-1] if k > 0 else -1
                    f_right = FORWARD_COUNT[k+1] if k < num_boundaries - 1 else -1
                    if FORWARD_COUNT[k] > f_left and FORWARD_COUNT[k] > f_right and FORWARD_COUNT[k] > 1:
                        forward_peaks.add(k)

                    b_left = BACKWARD_COUNT[k-1] if k > 0 else -1
                    b_right = BACKWARD_COUNT[k+1] if k < num_boundaries - 1 else -1
                    if BACKWARD_COUNT[k] > b_left and BACKWARD_COUNT[k] > b_right and BACKWARD_COUNT[k] > 1:
                        backward_peaks.add(k)

                if strategy == "union":
                    cut_positions = forward_peaks.union(backward_peaks)
                else:
                    cut_positions = forward_peaks.intersection(backward_peaks)

            segments = []
            start_idx = 0
            for k in range(num_boundaries):
                if k in cut_positions:
                    segments.append(word[start_idx:k+1])
                    start_idx = k + 1
            segments.append(word[start_idx:])

            segmented_words.append("-".join(segments))

        morph.append(" ".join(segmented_words))

    return morph
