import ast
import os
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

script_dir = Path(__file__).resolve().parent.parent
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

BANTU_CORPUS_MAP = {
    'ganda': {
        'path': "michsethowusu/english-ganda_sentence-pairs",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (str(row['Ganda']).lower(), str(row['English']))
    },
    'gikuyu': {
        'path': "michsethowusu/english-kikuyu_sentence-pairs",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (str(row['Kikuyu']).lower(), str(row['English']))
    },
    'tshiluba': {
        'path': "michsethowusu/english-tshiluba_sentence-pairs",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (str(row['Tshiluba']).lower(), str(row['English']))
    },
    'chiga': {
        'path': "michsethowusu/Code-170k-kiga",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (
            str(row['conversations'][0]['value']).lower() if isinstance(row['conversations'], list)
            else str(row['kiga_text']).lower(),
            str(row['english_translation'])
        )
    },
    'tooro': {
        'path': "michsethowusu/english-tooro_sentence-pairs",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (str(row['Tooro']).lower(), str(row['English']))
    },
    'runyoro': {
        'path': "michsethowusu/english-nyoro_sentence-pairs",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (str(row['Nyoro']).lower(), str(row['English']))
    },
    'kamba': {
        'path': "michsethowusu/english-kamba_sentence-pairs",
        'name': None,
        'split': "train",
        'extract_fn': lambda row: (str(row['Kamba']).lower(), str(row['English']))
    }
}

bantu_iso_map = {
    'ganda': 'lug', 'gikuyu': 'kik', 'tshiluba': 'lua',
    'chiga': 'cgg', 'tooro': 'ttj', 'runyoro': 'nyo', 'kamba': 'kam'
}

def load_hf_corpus_context(lang: str) -> dict:
    cfg = BANTU_CORPUS_MAP.get(lang.lower())
    if not cfg:
        return {}
    corpus_words_map = {}
    try:
        if cfg['name']:
            ds = load_dataset(cfg['path'], name=cfg['name'], split=cfg['split'], token=TOKEN).to_pandas()
        else:
            ds = load_dataset(cfg['path'], split=cfg['split'], token=TOKEN).to_pandas()

        for _, row in ds.iterrows():
            try:
                bantu_sentence, eng_context = cfg['extract_fn'](row)
                clean_sentence = bantu_sentence.strip(".,;:!?()\\\"'-")
                for word in clean_sentence.split():
                    w = word.strip(".,;:!?()\\\"'-")
                    if len(w) > 2 and w not in corpus_words_map:
                        corpus_words_map[w] = f"[HF Context] {eng_context.strip()}"
            except Exception:
                continue
    except Exception as e:
        print(f"Skipping HF corpus mapping: {e}")
    return corpus_words_map

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
            if sub_glossed:
                glossed_parts.append("-".join(sub_glossed))
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

def normalize_ortho(word: str) -> str:
    w = str(word).lower().strip("[]'\\\" ")
    prefixes = ['umu', 'aba', 'oki', 'oku', 'emi', 'eki', 'aka', 'omu', 'en']
    for p in prefixes:
        if w.startswith(p) and len(w) > len(p) + 2:
            return w[len(p):]
    return w

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
    if not iso_code:
        return

    lang_data_df = get_lang_data(lang)
    lang_txt_col = f"txt_{iso_code}"
    hf_corpus_map = load_hf_corpus_context(lang)

    model_path = script_dir / f"{file_name}"
    if not model_path.exists():
        model_path = script_dir / f"{lang.lower()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    if 'surface_word' not in model_df.columns and len(model_df.columns) > 0:
        if 'Unnamed: 0' in model_df.columns:
            model_df.rename(columns={'Unnamed: 0': 'surface_word'}, inplace=True)
        else:
            model_df.rename(columns={model_df.columns[0]: 'surface_word'}, inplace=True)

    exact_translation_map = {}
    if not lang_data_df.empty:
        for _, row in lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).iterrows():
            dict_word = str(row[lang_txt_col]).lower().strip()
            eng_trans = str(row['txt_eng'])
            if dict_word not in exact_translation_map:
                exact_translation_map[dict_word] = eng_trans

    output_data = {
        'Surface Word': [],
        f'{lang.capitalize()} Lemma': [],
        'English translation': [],
        'Glossing': []
    }

    for _, row in model_df.iterrows():
        surface_word = str(row['surface_word'])

        raw_lemmas = ast.literal_eval(row['lemmatization'])
        predicted_lemma = str(raw_lemmas[0]).lower().strip() if raw_lemmas else surface_word.lower()
        normalized_lemma = normalize_ortho(predicted_lemma)

        affix = affix_translate(ast.literal_eval(row['segmentation']), lang)
        matched = False

        # Tier 1: Exact Match (Normalized vs Dictionary)
        for lookup_key in [normalized_lemma, predicted_lemma, surface_word.lower()]:
            if lookup_key in exact_translation_map:
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(lookup_key)
                output_data['English translation'].append(exact_translation_map[lookup_key])
                output_data['Glossing'].append(affix)
                matched = True
                break

        # Tier 2: Substring Stem Overlap (Catches morphologically complex lemmas)
        if not matched:
            for dict_word, eng_trans in exact_translation_map.items():
                if len(dict_word) > 2 and (dict_word in normalized_lemma or normalized_lemma in dict_word):
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                    output_data['English translation'].append(eng_trans)
                    output_data['Glossing'].append(affix)
                    matched = True
                    break

        # Tier 3: Contextual Sentence Text Search
        if not matched and hf_corpus_map:
            for word_key, context_sentence in hf_corpus_map.items():
                if normalized_lemma in word_key or word_key in normalized_lemma:
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(word_key)
                    output_data['English translation'].append(context_sentence)
                    output_data['Glossing'].append(affix)
                    matched = True
                    break

        # Tier 4: Fuzzy Match Fallback
        if not matched and exact_translation_map:
            matches = get_close_matches(normalized_lemma, exact_translation_map.keys(), n=1, cutoff=0.7)
            if matches:
                best_match = matches[0]
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(best_match)
                output_data['English translation'].append(exact_translation_map[best_match])
                output_data['Glossing'].append(affix)
                matched = True

    df_out = pd.DataFrame(output_data).drop_duplicates(subset=['Surface Word']).reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Processed {lang}. Output entries: {len(df_out)}")
