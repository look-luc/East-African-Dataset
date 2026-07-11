import ast
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from morpheme_translate.extraction import root_extract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent.parent
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

bantu_iso_map = {
    'ganda': 'lug', 'gikuyu': 'kik', 'tshiluba': 'lua',
    'chiga': 'cgg', 'tooro': 'ttj', 'runyoro': 'nyo', 'kamba': 'kam'
}

bantuberta_id = "dsfsi/BantuBERTa"  # Or specific language variations if available
bb_tokenizer = AutoTokenizer.from_pretrained(bantuberta_id)
bb_model = AutoModel.from_pretrained(bantuberta_id)
bb_model = bb_model.to(device)
bb_model.eval()

def get_bantuberta_embedding(sentence: str, target_word: str):
    """Generates a contextual embedding isolated for a specific word inside its proverb context."""
    inputs = {k: v.to(device) for k, v in bb_tokenizer(sentence, return_tensors="pt").items()}

    word_tokens = bb_tokenizer.tokenize(target_word)
    word_ids = bb_tokenizer.convert_tokens_to_ids(word_tokens)

    with torch.no_grad():
        outputs = bb_model(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0)

    input_ids = inputs["input_ids"].squeeze(0).tolist()
    match_indices = [i for i, idx in enumerate(input_ids) if idx in word_ids]

    if not match_indices:
        return torch.mean(hidden_states, dim=0)

    word_embeddings = hidden_states[match_indices]
    return torch.mean(word_embeddings, dim=0)

def build_model_glossary(lang:str):
    lang_folder = script_dir / "data" / lang.lower()
    df = pd.read_csv(f"{lang_folder}/{lang.lower().capitalize()}_model_lem_seg.csv", sep="\t")
    model_map = {}
    for _, row in df.iterrows():
        try:
            segments = ast.literal_eval(row['segmentation'])
            glosses = ast.literal_eval(row['noun class prediction'])

            for seg, gloss in zip(segments, glosses):
                model_map[str(seg).lower().strip()] = str(gloss).strip()
        except (ValueError, SyntaxError, TypeError):
            continue
    return model_map

grammar_df = pd.read_csv(str(script_dir / "data/bantu_grammar_lookup.csv"))

def affix_translate(segments, language, model_glossary):
    grammar = grammar_df[grammar_df['language'] == str(language).lower()]
    manual_map = {
        str(k).strip().lower(): str(v).strip()
        for k, v in zip(grammar['morpheme_segment'], grammar['proposed_leipzig_gloss'])
        if pd.notna(k) and pd.notna(v)
    }

    master_map = {**model_glossary, **manual_map}
    glossed_parts = []

    for seg in segments:
        clean_seg = str(seg).lower().strip()
        if clean_seg in master_map:
            glossed_parts.append(master_map[clean_seg])
        elif "-" in clean_seg:
            sub_parts = clean_seg.split("-")
            sub_glossed = []
            for part in sub_parts:
                p = part.strip()
                if p in master_map:
                    sub_glossed.append(master_map[p])
                else:
                    sub_glossed.append(f"[{p}]")
            glossed_parts.append("-".join(sub_glossed))
        else:
            glossed_parts.append(f"[{clean_seg}]")

    return "-".join(glossed_parts)

def normalize_ortho(word: str, lang: str = "") -> str:
    w = str(word).lower().strip("[]'\\\" ")
    lang = lang.lower()

    if lang in ['ganda', 'gikuyu', 'chiga', 'tooro', 'runyoro', 'kamba']:
        prefixes = ['umu', 'aba', 'oki', 'oku', 'emi', 'eki', 'aka', 'omu', 'en']
        for p in prefixes:
            if w.startswith(p) and len(w) >= len(p) + 2:
                return w[len(p):]

    elif lang == 'tshiluba':
        tshiluba_prefixes = ['tshi', 'bi', 'mu', 'ba', 'di', 'ma', 'lu', 'ka', 'tu']
        for p in tshiluba_prefixes:
            if w.startswith(p) and len(w) >= len(p) + 2:
                return w[len(p):]

    return w

def strip_accents(text: str) -> str:
    """Removes tonal accent marks (like à, ê) to align PanLex with model lemmas."""
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

@lru_cache(maxsize=64)
def get_lang_data(lang: str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        print(f"Warning: {lang} is not in the target Bantu dictionary.")
        return pd.DataFrame()

    panlex_data = load_dataset("cointegrated/panlex-meanings", name=iso_code, split="train").select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()

    eng_data = load_dataset("cointegrated/panlex-meanings", name='eng', split="train").select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()
    df_eng = eng_data[eng_data['langvar_uid'].str.startswith('eng', na=False)]

    try:
        eng_def_data = load_dataset("cointegrated/panlex-definitions", name='eng', split="train").select_columns(["meaning", "txt", "langvar_uid"]).to_pandas()
        df_eng_def = eng_def_data[eng_def_data['langvar_uid'].str.startswith('eng', na=False)]
        df_eng = pd.concat([df_eng, df_eng_def], ignore_index=True)
    except Exception as e:
        print(f"Note: Could not supplement with panlex-definitions: {e}")

    panlex_df = panlex_data.merge(
        df_eng, on='meaning', how='inner', suffixes=(f'_{iso_code}', '_eng')
    )

    noise = ['dollar', 'pound', 'shilling', 'republic', 'ocean', 'sea', 'continent', 'st.', 'saudi', 'papua', 'zimbabwe', 'sudanese']
    noise_regex = '|'.join([rf'\b{w}\b' for w in noise])
    panlex_df = panlex_df[~panlex_df['txt_eng'].str.contains(noise_regex, case=False, na=False)]
    panlex_df = panlex_df[panlex_df['txt_eng'].str.len() < 100]

    panlex_df['txt_eng'] = panlex_df['txt_eng'].str.split(r':|/').str[0].str.strip()

    target_word_col = f'txt_{iso_code}'

    panlex_df = panlex_df.dropna(subset=[target_word_col, 'txt_eng'])
    panlex_df = panlex_df.groupby(target_word_col)['txt_eng'].apply(
        lambda x: ", ".join(dict.fromkeys(x.dropna().astype(str)))
    ).reset_index()

    return panlex_df

def translation(file_name: str, lang: str, local_proverbs_title: str|None = None):
    lang_folder = script_dir / "data" / lang.lower()

    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        return

    lang_data_df = get_lang_data(lang)
    lang_txt_col = f"txt_{iso_code}"

    model_path = lang_folder / f"{file_name}"
    if not model_path.exists():
        model_path = lang_folder / f"{lang.lower().capitalize()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    if 'surface_word' not in model_df.columns and len(model_df.columns) > 0:
        if 'Unnamed: 0' in model_df.columns:
            model_df.rename(columns={'Unnamed: 0': 'surface_word'}, inplace=True)
        else:
            model_df.rename(columns={model_df.columns[0]: 'surface_word'}, inplace=True)

    lang_grammar = grammar_df[grammar_df['language'] == str(lang).lower()]
    valid_grammar = lang_grammar.dropna(subset=['proposed_leipzig_gloss'])

    known_affixes = set(
        valid_grammar['morpheme_segment']
        .str.replace('-', '', regex=False)
        .str.lower()
        .dropna()
    )

    exact_translation_map = {}

    if not lang_data_df.empty:
        valid_pairs = lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).copy()
        valid_pairs['clean_key'] = valid_pairs[lang_txt_col].astype(str).apply(strip_accents).str.lower().str.strip()

        valid_pairs = valid_pairs.groupby('clean_key')['txt_eng'].apply(
            lambda x: ", ".join(dict.fromkeys(x.dropna().astype(str)))
        ).reset_index()

        exact_translation_map = dict(zip(valid_pairs['clean_key'], valid_pairs['txt_eng']))

    print("=" * 60)
    print(f"DIAGNOSTIC LOG FOR: {lang.upper()}")
    print(f"1. Total raw panlex rows fetched: {len(lang_data_df)}")
    print(f"2. Size of built exact_translation_map: {len(exact_translation_map)}")
    if exact_translation_map:
        print(f"3. Sample dictionary keys (first 15): {list(exact_translation_map.keys())[:15]}")
    else:
        print("3. Sample dictionary keys: [NONE - DICTIONARY IS EMPTY]")
    print("=" * 60)

    local_proverb_sentences = []
    if local_proverbs_title:
        proverbs_path = script_dir / local_proverbs_title
        if proverbs_path.exists():
            prov_df = pd.read_csv(str(proverbs_path), sep='\t')
            if 'language' in prov_df.columns and 'african_proverb' in prov_df.columns:
                lang_prov_df = prov_df[prov_df["language"].str.lower() == lang.lower()]
                local_proverb_sentences = lang_prov_df["african_proverb"].dropna().tolist()

    output_data = {
        'Surface Word': [],
        f'{lang.capitalize()} Lemma': [],
        'English translation': [],
        'Glossing': [],
        'Match Type': []
    }
    successful_embeddings_pool = {}
    model_glossary = build_model_glossary(lang)
    for _, row in model_df.iterrows():
        surface_word = str(row['surface_word'])
        raw_lemmas = ast.literal_eval(row['lemmatization'])
        predicted_lemma = str(raw_lemmas[0]).lower().strip() if raw_lemmas else surface_word.lower()
        normalized_lemma = normalize_ortho(predicted_lemma)
        segmentation_str = row['segmentation']
        affix = affix_translate(ast.literal_eval(segmentation_str), lang, model_glossary)

        matched = False

        clean_lemma_token = normalized_lemma.lower().replace('-', '').strip()
        is_standalone_grammar = clean_lemma_token in known_affixes

        # Tier 1: Exact Match
        for lookup_key in [normalized_lemma, predicted_lemma, surface_word.lower()]:
            clean_lookup = strip_accents(lookup_key).lower().strip()
            if clean_lookup in exact_translation_map:
                translation_text = exact_translation_map[clean_lookup]
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(lookup_key)
                output_data['English translation'].append(translation_text)
                output_data['Glossing'].append(affix)
                output_data['Match Type'].append('PanLex Exact Match')
                matched = True
                break

        # Tier 2: Substring Stem Overlap
        if not matched and not is_standalone_grammar:
            if len(normalized_lemma) >= 4:
                clean_norm_lemma = strip_accents(normalized_lemma).lower().strip()
                for dict_word, eng_trans in exact_translation_map.items():
                    if len(dict_word) > 3 and (dict_word in clean_norm_lemma or clean_norm_lemma in dict_word):
                        translation_text = eng_trans
                        output_data['Surface Word'].append(surface_word)
                        output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                        output_data['English translation'].append(eng_trans)
                        output_data['Glossing'].append(affix)
                        output_data['Match Type'].append('Substring Overlap')
                        matched = True
                        break

        # Tier 2 Fallback: Grammar Guard
        if not matched and is_standalone_grammar:
            noun_class = None
            raw_nc = ast.literal_eval(row['noun class prediction'])
            if raw_nc:
                nc_string = str(raw_nc[0])
                features = nc_string.split()
                if len(features) > 1:
                    tags = features[1].split(';')
                    predicted_tag = tags[-1]
                    if "BANTU" in predicted_tag or "NC" in predicted_tag:
                        noun_class = predicted_tag

            if noun_class:
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(f"(-){surface_word}(-)")
                output_data['English translation'].append(noun_class)
                output_data['Glossing'].append(affix)
                output_data['Match Type'].append('Grammar Guard Fallback')
                matched = True

        # Tier 3: Isolated Local Root Evaluation
        if not matched:
            extracted_roots = root_extract(segmentation_str, lang)
            for root in extracted_roots:
                clean_root = strip_accents(root).lower().strip()
                if clean_root in exact_translation_map:
                    translation_text = exact_translation_map[clean_root]
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(root)
                    output_data['English translation'].append(translation_text)
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('Isolated Root Exact Match')
                    matched = True
                    break

        # ---- POPULATE EMBEDDING POOL UPON SUCCESS ----
        if matched and local_proverb_sentences:
            for proverb in local_proverb_sentences:
                if surface_word in proverb:
                    matched_vec = get_bantuberta_embedding(proverb, surface_word)
                    successful_embeddings_pool[matched_vec] = translation_text
                    break

        # Tier 4: Word-Bounded Local Proverb Context Check
        if not matched and local_proverb_sentences:
            current_proverb = ""
            for proverb in local_proverb_sentences:
                if surface_word in proverb:
                    current_proverb = proverb
                    break

            if current_proverb and successful_embeddings_pool:  # Guard against empty pools
                unknown_vec = get_bantuberta_embedding(current_proverb, surface_word)

                best_score = -1.0
                best_translation = None

                for known_vec, known_translation in successful_embeddings_pool.items():
                    similarity = F.cosine_similarity(unknown_vec.unsqueeze(0), known_vec.unsqueeze(0)).item()
                    if similarity > best_score:
                        best_score = similarity
                        best_translation = known_translation

                if best_score > 0.82:
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(predicted_lemma)
                    output_data['English translation'].append(f"{best_translation} (Inferred via BantuBERTa)")
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('BantuBERTa Vector Neighborhood')
                    matched = True

        # Fallback when no translation tier catches it
        if not matched:
            output_data['Surface Word'].append(surface_word)
            output_data[f'{lang.capitalize()} Lemma'].append(predicted_lemma)
            output_data['English translation'].append('[Translation Missing]')
            output_data['Glossing'].append(affix)
            output_data['Match Type'].append('No Lexicon Match')

    df_out = pd.DataFrame(output_data).drop_duplicates(subset=['Surface Word']).reset_index(drop=True)
    df_out.to_csv(lang_folder / f"{lang.lower()}_translated.csv", index=False)
    print(f"Processed {lang}. Output entries: {len(df_out)}")
