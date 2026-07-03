import ast
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset

script_dir = Path(__file__).resolve().parent.parent

bantu_iso_map = {
    'nyaturu': 'rim', 'bangubangu': 'bby', 'kwele': 'kwl', 'kihangaza': 'han',
    'soga': 'xog', 'pare': 'asa', 'olusamia': 'lsm', 'taabwa': 'tap',
    'nande': 'nnb', 'hemba': 'hem', 'tshiluba': 'lua', 'tooro': 'ttj',
    'hema': 'hea', 'holoholo': 'hoo', 'meru': 'mer', 'runyoro': 'nyo',
    'zigula': 'ziw', 'makonde': 'kde', 'kamba': 'kam', 'digo': 'dig',
    'kihara': 'haq', 'nyala': 'nle', 'gikuyu': 'kik', 'tetela': 'tll',
    'rufumbira': 'kin', 'sukuma': 'suk', 'ganda': 'lug', 'gweno': 'gwe',
    'chiga': 'cgg', 'ekegusii': 'guz'
}

def affix_translate(segments, language):
    grammar_df = pd.read_csv(str(script_dir / "bantu_grammar_lookup.csv"))
    grammar = grammar_df[grammar_df['language'] == str(language).lower()]
    grammar_map = dict(zip(grammar['morpheme_segment'], grammar['proposed_leipzig_gloss']))

    glossed_parts = []
    for seg in segments:
        clean_seg = seg.lower().strip('-')
        if clean_seg in grammar_map:
            glossed_parts.append(str(grammar_map[clean_seg]))
        else:
            glossed_parts.append(str(clean_seg))
    return "-".join(glossed_parts)

@lru_cache(maxsize=64)
def get_lang_data(lang:str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        print(f"Warning: {lang} is not in the target Bantu dictionary.")
        return pd.DataFrame()

    panlex_data = load_dataset("cointegrated/panlex-meanings", name=iso_code, split="train") \
        .select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()

    eng_data = load_dataset("cointegrated/panlex-meanings", name='eng', split="train") \
        .select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()

    panlex_df = panlex_data.merge(
        eng_data, on='meaning', how='left', suffixes=(f'_{iso_code}', '_eng')
    ).drop_duplicates(subset=[f'txt_{iso_code}', 'txt_eng'])

    noise = ['dollar', 'pound', 'shilling', 'republic', 'ocean', 'sea', 'continent', 'st.', 'saudi', 'papua', 'zimbabwe', 'sudanese']
    noise_regex = '|'.join(noise)
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(noise_regex, case=False, na=False)]
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(r':|/', na=False)]
    panlex_df = panlex_df[panlex_df['txt_eng'].str.len() < 100]
    panlex_df = panlex_df[panlex_df['langvar_uid_eng'] == 'eng-000']
    return panlex_df

def normalize_ortho(lemma: str):
    lemma = lemma.strip().lower().strip('-')
    if lemma.startswith('um'):
        lemma = 'om' + lemma[2:]
    elif lemma.startswith('un'):
        lemma = 'on' + lemma[2:]
    elif lemma.startswith('in'):
        lemma = 'en' + lemma[2:]
    elif lemma.startswith('im'):
        lemma = 'em' + lemma[2:]
    return lemma

def strip_bantu_prefixes(word: str) -> str:
    """
    Strips common Luganda nominal augments, noun class prefixes,
    and verbal infinitive markers to expose the core stem for matching.
    """
    word = word.strip().lower().strip('-')

    prefixes = [
        'omw', 'aba', 'amy', 'emi', 'eri', 'ama', 'eki', 'ebi', 'eji', 'aka', 'otu', 'olu', 'ens', 'obu', 'oku', 'ogu', 'egi',
        'omu', 'umu', 'aba', 'imi', 'iki', 'ibi', 'aka', 'ulu', 'ubu', 'uku',
        'mu', 'ba', 'mi', 'li', 'ma', 'ki', 'bi', 'ka', 'tu', 'lu', 'bu', 'ku', 'gu', 'gi', 'n', 'm'
    ]
    for p in sorted(prefixes, key=len, reverse=True):
        if word.startswith(p) and len(word) > len(p):
            return word[len(p):]
    return word

def translation(file_name: str, lang: str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        print(f"Error: {lang} mapping not found.")
        return

    lang_data_df = get_lang_data(lang)
    lang_txt_col = f"txt_{iso_code}"

    model_path = script_dir / f"{file_name}"
    if not model_path.exists():
        model_path = script_dir / f"{lang.lower()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    # Build primary dictionary map and secondary stem-optimized map
    exact_translation_map = {}
    stem_translation_map = {}

    for _, row in lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).iterrows():
        clean_dict_key = str(row[lang_txt_col]).strip().lower().strip('-')
        translation_val = str(row['txt_eng'])

        exact_translation_map[clean_dict_key] = translation_val

        # Populate stem map for cross-prefix fallback
        dict_stem = strip_bantu_prefixes(clean_dict_key)
        if len(dict_stem) >= 2 and dict_stem not in stem_translation_map:
            stem_translation_map[dict_stem] = (clean_dict_key, translation_val)

    if model_df.columns[0] == 'Unnamed: 0':
        model_df.rename(columns={'Unnamed: 0': 'surface_word'}, inplace=True)
    else:
        model_df.index.name = 'surface_word'
        model_df = model_df.reset_index()

    output_data = {
        'Surface Word': [],
        f'{lang.capitalize()} Lemma': [],
        'English translation': [],
        'Glossing': []
    }

    for _, row in model_df.iterrows():
        surface_word = str(row['surface_word'])
        raw_lemmas = row['lemmatization']

        try:
            lemma_list = ast.literal_eval(raw_lemmas)
            if not isinstance(lemma_list, list) or not lemma_list:
                continue
            model_lemma = str(lemma_list[0])

            segments = ast.literal_eval(row['segmentation'])
            affix = affix_translate(segments, lang)
        except (ValueError, SyntaxError, IndexError):
            continue

        normalized_lemma = normalize_ortho(model_lemma)
        matched = False

        # Tier 1: Exact Match
        if normalized_lemma in exact_translation_map:
            output_data['Surface Word'].append(surface_word)
            output_data[f'{lang.capitalize()} Lemma'].append(normalized_lemma)
            output_data['English translation'].append(exact_translation_map[normalized_lemma])
            output_data['Glossing'].append(affix)
            matched = True

        # Tier 2: Stem-to-Stem Match (Matches variations of noun prefixes)
        if not matched:
            lemma_stem = strip_bantu_prefixes(normalized_lemma)
            if lemma_stem in stem_translation_map:
                dict_word, eng_trans = stem_translation_map[lemma_stem]
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                output_data['English translation'].append(eng_trans)
                output_data['Glossing'].append(affix)
                matched = True

        # Tier 3: Verbal/Infinitive Fallback Check (.endswith)
        if not matched and len(normalized_lemma) > 2:
            for dict_word, eng_trans in exact_translation_map.items():
                if dict_word.endswith(normalized_lemma) and len(dict_word) <= len(normalized_lemma) + 3:
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                    output_data['English translation'].append(eng_trans)
                    output_data['Glossing'].append(affix)
                    matched = True
                    break
                elif normalized_lemma.endswith(dict_word) and len(normalized_lemma) <= len(dict_word) + 3:
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                    output_data['English translation'].append(eng_trans)
                    output_data['Glossing'].append(affix)
                    matched = True
                    break

        # Tier 4: Glossing Preservation Fallback (Crucial for manual lookup workflows)
        if not matched:
            output_data['Surface Word'].append(surface_word)
            output_data[f'{lang.capitalize()} Lemma'].append(model_lemma)
            output_data['English translation'].append('[UNKNOWN]') # Keeps row intact for manual annotation
            output_data['Glossing'].append(affix)

    df_out = pd.DataFrame(output_data).drop_duplicates().reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Successfully processed model data. Saved entries to {lang.lower()}_translated.csv with {len(df_out)} unique pairs.")
