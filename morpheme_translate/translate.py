import ast
from difflib import get_close_matches
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

flores_code = {
    'lug': 'lug_Latn', 'kik': 'kik_Latn', 'kam': 'kam_Latn',
    'suk': 'suk_Latn', 'lua': 'lua_Latn', 'kin': 'kin_Latn', 'mer': 'mer_Latn',

    'xog': 'lug_Latn',
    'nyo': 'nyo_Latn',
    'ttj': 'nyo_Latn',
    'cgg': 'nyo_Latn',
    'lsm': 'luy_Latn',
    'nle': 'luy_Latn',

    'asa': 'swh_Latn',
    'gwe': 'swh_Latn',
    'dig': 'swh_Latn',
    'ziw': 'swh_Latn',
}

def flores_bantu(iso_code:str):
    target_iso = flores_code.get(iso_code, iso_code)
    flores_words_map = {}

    try:
        flores_ds = load_dataset("facebook/flores", name=target_iso, split="dev").to_pandas()
        eng_flores = load_dataset("facebook/flores", name="eng_Latn", split="dev").to_pandas()

        for idx, row in flores_ds.iterrows():
            target_sentence = str(row['sentence']).lower()
            eng_sentence = str(eng_flores.iloc[idx]['sentence']).strip()

            for word in target_sentence.split():
                clean_word = word.strip(".,;:!?()\"'-")
                if clean_word and clean_word not in flores_words_map:
                    # Map the isolated token to its parallel translation context
                    flores_words_map[clean_word] = f"[FLORES Context] {eng_sentence}"
    except Exception as e:
        print(f"Notice: FLORES mapping skipped for config '{target_iso}'. Reason: {e}")
        return {}

    return flores_words_map

grammar_df = pd.read_csv(str(script_dir / "bantu_grammar_lookup.csv"))
def affix_translate(segments, language):
    grammar = grammar_df[grammar_df['language'] == str(language).lower()]
    grammar_map = dict(zip(grammar['morpheme_segment'], grammar['proposed_leipzig_gloss']))

    glossed_parts = []
    for seg in segments:
        clean_seg = seg.lower()
        if "-" in clean_seg and clean_seg not in grammar_map:
            sub_parts = clean_seg.split("-")
            sub_glossed = []

            for i, part in enumerate(sub_parts):
                if not part:
                    continue

                prefix_candidate = part + '-'
                suffix_candidate = '-' + part

                if prefix_candidate in grammar_map:
                    sub_glossed.append(str(grammar_map[prefix_candidate]))
                elif suffix_candidate in grammar_map:
                    sub_glossed.append(str(grammar_map[suffix_candidate]))
                elif part in grammar_map:
                    sub_glossed.append(str(grammar_map[part]))
                else:
                    sub_glossed.append(part)
        else:
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

    panlex_data = load_dataset("cointegrated/panlex-meanings", name=iso_code, split="train").select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()

    eng_data = load_dataset("cointegrated/panlex-meanings", name='eng', split="train").select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()
    eng_data = eng_data[eng_data['langvar_uid'] == 'eng-000']

    ds_eng_word = load_dataset('cointegrated/panlex-definitions', name='eng', split='train').select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()
    ds_eng_word = ds_eng_word[ds_eng_word['langvar_uid'] == 'eng-000'].rename(columns={'txt': 'definition_text'})

    df_eng = eng_data.merge(ds_eng_word, on='meaning', how='left')
    df_eng = df_eng.drop_duplicates(subset=['txt', 'definition_text'])

    panlex_df = panlex_data.merge(
        df_eng, on='meaning', how='left', suffixes=(f'_{iso_code}', '_eng')
    ).drop_duplicates(subset=[f'txt_{iso_code}', 'txt_eng'])

    noise = ['dollar', 'pound', 'shilling', 'republic', 'ocean', 'sea', 'continent', 'st.', 'saudi', 'papua', 'zimbabwe', 'sudanese']
    noise_regex = '|'.join(noise)
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(noise_regex, case=False, na=False)]
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(r':|/', na=False)]
    panlex_df = panlex_df[panlex_df['txt_eng'].str.len() < 100]

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
        "umu", "um"
    ]
    for p in sorted(prefixes, key=len, reverse=True):
        if word.startswith(p) and len(word) > len(p):
            return word[len(p):]
    return word

def translation(file_name: str, lang: str):
    iso_code = bantu_iso_map.get(lang.lower())
    flores_map = flores_bantu(iso_code)

    if not iso_code:
        print(f"Error: {lang} mapping not found.")
        return

    lang_data_df = get_lang_data(lang)
    lang_txt_col = f"txt_{iso_code}"

    model_path = script_dir / f"{file_name}"
    if not model_path.exists():
        model_path = script_dir / f"{lang.lower()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    exact_translation_map = {}
    stem_translation_map = {}

    for _, row in lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).iterrows():
        clean_dict_key = str(row[lang_txt_col]).strip().lower().strip('-')

        eng_word = str(row['txt_eng']).strip()
        def_text = str(row['definition_text']).strip() if pd.notna(row['definition_text']) else ""

        if def_text and def_text.lower() != 'none':
            translation_val = f"{eng_word} (Context: {def_text})"
        else:
            translation_val = eng_word

        exact_translation_map[clean_dict_key] = translation_val

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
                # FLORES logic
                if not matched and flores_map:
                    if normalized_lemma in flores_map:
                        output_data['Surface Word'].append(surface_word)
                        output_data[f'{lang.capitalize()} Lemma'].append(normalized_lemma)
                        output_data['English translation'].append(flores_map[normalized_lemma])
                        output_data['Glossing'].append(affix)
                        matched = True
                        break


        # Tier 4: Glossing Preservation Fallback (Crucial for manual lookup workflows)
        if not matched:
            matches = get_close_matches(normalized_lemma, exact_translation_map.keys(), n=1, cutoff=0.8)

            if matches:
                best_match = matches[0]
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(best_match)
                output_data['English translation'].append(exact_translation_map[best_match])
                output_data['Glossing'].append(affix)
                matched = True

    df_out = pd.DataFrame(output_data).drop_duplicates().reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Successfully processed model data. Saved entries to {lang.lower()}_translated.csv with {len(df_out)} unique pairs.")
