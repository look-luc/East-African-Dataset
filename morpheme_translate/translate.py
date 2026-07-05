import ast
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

script_dir = Path(__file__).resolve().parent.parent
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

bantu_iso_map = {
    'ganda': 'lug', 'gikuyu': 'kik', 'tshiluba': 'lua',
    'chiga': 'cgg', 'tooro': 'ttj', 'runyoro': 'nyo', 'kamba': 'kam'
}

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

def translation(file_name: str, lang: str):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        return

    lang_data_df = get_lang_data(lang)
    lang_txt_col = f"txt_{iso_code}"

    model_path = script_dir / f"{file_name}"
    if not model_path.exists():
        model_path = script_dir / f"{lang.lower()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    if 'surface_word' not in model_df.columns and len(model_df.columns) > 0:
        if 'Unnamed: 0' in model_df.columns:
            model_df.rename(columns={'Unnamed: 0': 'surface_word'}, inplace=True)
        else:
            model_df.rename(columns={model_df.columns[0]: 'surface_word'}, inplace=True)

    # --- VECTOR PREPARATION FOR PANLEX DICTIONARY ---
    from model_segment import get_word_embedding

    panlex_words = []
    panlex_translations = []
    panlex_vectors = []

    print("Vectorizing PanLex dictionary references...")
    if not lang_data_df.empty:
        unique_pairs = lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).drop_duplicates(subset=[lang_txt_col])
        for _, row in unique_pairs.iterrows():
            dict_word = str(row[lang_txt_col]).lower().strip()
            eng_trans = str(row['txt_eng'])

            panlex_words.append(dict_word)
            panlex_translations.append(eng_trans)
            panlex_vectors.append(get_word_embedding(dict_word))

    if panlex_vectors:
        panlex_tensor = torch.tensor(panlex_vectors) # Shape: [Num_PanLex_Words, Dimension]
        panlex_tensor = panlex_tensor / panlex_tensor.norm(dim=1, keepdim=True) # Normalize for Cosine Similarity
    else:
        panlex_tensor = None

    exact_translation_map = dict(zip(panlex_words, panlex_translations))

    output_data = {
        'Surface Word': [],
        f'{lang.capitalize()} Lemma': [],
        'English translation': [],
        'Glossing': [],
        'Match Type': []
    }

    for _, row in model_df.iterrows():
        surface_word = str(row['surface_word'])
        raw_lemmas = ast.literal_eval(row['lemmatization'])
        predicted_lemma = str(raw_lemmas[0]).lower().strip() if raw_lemmas else surface_word.lower()
        normalized_lemma = normalize_ortho(predicted_lemma)
        affix = affix_translate(ast.literal_eval(row['segmentation']), lang)

        matched = False

        # Tier 1: Exact Match
        for lookup_key in [normalized_lemma, predicted_lemma, surface_word.lower()]:
            if lookup_key in exact_translation_map:
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(lookup_key)
                output_data['English translation'].append(exact_translation_map[lookup_key])
                output_data['Glossing'].append(affix)
                output_data['Match Type'].append('Exact Match')
                matched = True
                break

        # Tier 2: Substring Stem Overlap
        if not matched:
            for dict_word, eng_trans in exact_translation_map.items():
                if len(dict_word) > 2 and (dict_word in normalized_lemma or normalized_lemma in dict_word):
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                    output_data['English translation'].append(eng_trans)
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('Substring Overlap')
                    matched = True
                    break

        # Tier 3: Vector Embedding Nearest Neighbor Search (Replaces Fuzzy Fallback)
        if not matched and panlex_tensor is not None and 'embedding' in row:
            try:
                query_vec = ast.literal_eval(row['embedding'])
                query_tensor = torch.tensor(query_vec).unsqueeze(0)
                query_tensor = query_tensor / query_tensor.norm(dim=1, keepdim=True)

                similarities = torch.mm(query_tensor, panlex_tensor.T).squeeze(0)
                best_match_idx = torch.argmax(similarities).item()
                highest_score = similarities[best_match_idx].item()

                if highest_score > 0.65:
                    best_word = panlex_words[best_match_idx]
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(best_word)
                    output_data['English translation'].append(panlex_translations[best_match_idx])
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append(f'Vector Search (Score: {highest_score:.2f})')
                    matched = True
            except Exception as e:
                print(f"Embedding resolution failed for {surface_word}: {e}")

    df_out = pd.DataFrame(output_data).drop_duplicates(subset=['Surface Word']).reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Processed {lang}. Output entries: {len(df_out)}")
