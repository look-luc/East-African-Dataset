import re
from collections import defaultdict

import pandas as pd
from pandas import DataFrame

HAS_AUGMENT = {
    'ganda': True, 'chiga': True, 'tooro': True, 'runyoro': True,
    'gikuyu': False, 'kamba': False, 'tshiluba': False
}

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

def clean_segments(segmented_word: str):
    # Bantu prefixes are typically V or CV, never standalone single obstruents
    segmented_word = re.sub(r'\b([aeioubcdfghjklmnpqrstvwxyz])-', r'\1', segmented_word)

    # Nasal+Consonant complexes like nt, nd, mp, mb act as single phonological units.
    segmented_word = re.sub(r'-([nt|nd|mp|mb|ng|nj])', r'\1', segmented_word)


def segment(df: DataFrame, min_word_len=4, product_threshold=15, ratio=0.35):
    """
    Segments words using the product of forward and backward variety scores,
    filtering out noise using an adjustable baseline threshold floor.
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
            # Length guardrail protects very small words/particles from being touched
            if len(word) < min_word_len:
                segmented_words.append(word)
                continue

            num_boundaries = len(word) - 1
            FORWARD_COUNT = []
            BACKWARD_COUNT = []

            for k in range(num_boundaries):
                prefix = word[:k+1]
                suffix_rev = word[k+1:][::-1]

                FORWARD_COUNT.append(len(forward_trie.get(prefix, set())))
                BACKWARD_COUNT.append(len(backward_trie.get(suffix_rev, set())))

            scores = [FORWARD_COUNT[k] * BACKWARD_COUNT[k] for k in range(num_boundaries)]
            max_word_peak = max(scores) if scores else 0
            dynamic_threshold = max(product_threshold, max_word_peak * ratio)

            cut_positions = set()

            for k in range(num_boundaries):
                left_val = scores[k-1] if k > 0 else -1
                right_val = scores[k+1] if k < num_boundaries - 1 else -1

                if scores[k] > left_val and scores[k] > right_val and scores[k] >= dynamic_threshold:
                    cut_positions.add(k)

            segments = []
            start_idx = 0
            for k in range(num_boundaries):
                if k in cut_positions:
                    segments.append(word[start_idx:k+1])
                    start_idx = k + 1
            segments.append(word[start_idx:])

            raw_segmented = "-".join(segments)
            final_segmented = clean_segments(raw_segmented)
            segmented_words.append(final_segmented)

        morph.append(" ".join(segmented_words))

    return morph
