import ast
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from morpheme_translate.extraction import root_extract

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

def normalize_ortho(word: str) -> str:
    w = str(word).lower().strip("[]'\\\" ")
    prefixes = ['umu', 'aba', 'oki', 'oku', 'emi', 'eki', 'aka', 'omu', 'en']
    for p in prefixes:
        if w.startswith(p) and len(w) > len(p) + 2:
            return w[len(p):]
    return w

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

def translation(file_name: str, lang: str, local_proverbs_title: str|None = None):
    iso_code = bantu_iso_map.get(lang.lower())
    if not iso_code:
        return

    lang_data_df = get_lang_data(lang)
    lang_txt_col = f"txt_{iso_code}"

    model_path = script_dir / f"{file_name}"
    if not model_path.exists():
        model_path = script_dir / f"{lang.lower().capitalize()}_model_lem_seg.csv"

    model_df = pd.read_csv(str(model_path), sep='\t')

    if 'surface_word' not in model_df.columns and len(model_df.columns) > 0:
        if 'Unnamed: 0' in model_df.columns:
            model_df.rename(columns={'Unnamed: 0': 'surface_word'}, inplace=True)
        else:
            model_df.rename(columns={model_df.columns[0]: 'surface_word'}, inplace=True)

    # Side A Grammar Guard Setup: Extract and clean known functional affixes to protect Tier 2
    lang_grammar = grammar_df[grammar_df['language'] == str(lang).lower()]
    known_affixes = set(
        lang_grammar['morpheme_segment']
        .str.replace('-', '', regex=False)
        .str.lower()
        .dropna()
    )

    panlex_words = []
    panlex_translations = []

    if not lang_data_df.empty:
        unique_pairs = lang_data_df.dropna(subset=[lang_txt_col, 'txt_eng']).drop_duplicates(subset=[lang_txt_col])
        for _, row in unique_pairs.iterrows():
            dict_word = str(row[lang_txt_col]).lower().strip()
            eng_trans = str(row['txt_eng'])
            panlex_words.append(dict_word)
            panlex_translations.append(eng_trans)

    exact_translation_map = dict(zip(panlex_words, panlex_translations))

    # Optional: Load local proverbs context mapping if a local source path is provided
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

    for _, row in model_df.iterrows():
        surface_word = str(row['surface_word'])
        raw_lemmas = ast.literal_eval(row['lemmatization'])
        predicted_lemma = str(raw_lemmas[0]).lower().strip() if raw_lemmas else surface_word.lower()
        normalized_lemma = normalize_ortho(predicted_lemma)
        segmentation_str = row['segmentation']
        affix = affix_translate(ast.literal_eval(segmentation_str), lang)

        matched = False

        # Tier 1: Exact Match (PanLex Lexicon Gate)
        for lookup_key in [normalized_lemma, predicted_lemma, surface_word.lower()]:
            if lookup_key in exact_translation_map:
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(lookup_key)
                output_data['English translation'].append(exact_translation_map[lookup_key])
                output_data['Glossing'].append(affix)
                output_data['Match Type'].append('PanLex Exact Match')
                matched = True
                break

        try:
            morpheme_segments = ast.literal_eval(str(row['segmentation']))  # e.g., ['aba-', 'kadde']
        except (ValueError, SyntaxError):
            morpheme_segments = []

        has_affix = False
        for segment in morpheme_segments:
            clean_segment = str(segment).lower().replace('-', '').strip()

            if clean_segment in known_affixes:
                has_affix = True
                break

        # Tier 2: Substring Stem Overlap (Lexicon Containment Gate)
        if not matched and not has_affix:
            for dict_word, eng_trans in exact_translation_map.items():
                if len(dict_word) > 4 and (dict_word in normalized_lemma or normalized_lemma in dict_word):
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(dict_word)
                    output_data['English translation'].append(eng_trans)
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('Substring Overlap')
                    matched = True
                    break

        if not matched and has_affix:
            for affix in known_affixes:
                if affix in model_path["lemmatization"]:
                    noun_class = None

                    # Safely parse noun class string and extract class tag
                    raw_nc = ast.literal_eval(row['noun class prediction'])
                    if raw_nc:
                        nc_string = str(raw_nc[0])  # Fixed Indexing Bug (changed i to 0)
                        features = nc_string.split()
                        if len(features) > 1:
                            tags = features[1].split(';')  # ['N', 'PL', 'BANTU2']
                            noun_class = tags[-1]          # "BANTU2"
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(f"(-){surface_word}(-)")
                    output_data['English translation'].append(noun_class)
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('Substring Overlap')
                    matched = True
                    break

        # Tier 3: Isolated Local Root Evaluation (Leveraging local grammar strip rules)
        if not matched:
            extracted_roots = root_extract(segmentation_str)
            for root in extracted_roots:
                clean_root = root.lower().strip()
                if clean_root in exact_translation_map:
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(clean_root)
                    output_data['English translation'].append(exact_translation_map[clean_root])
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('Isolated Root Exact Match')
                    matched = True
                    break

        # Alternative Tier 4: Word-Bounded Local Proverb Context Check (Only runs if text path is loaded)
        if not matched and local_proverb_sentences:
            pattern = re.compile(r'\b' + re.escape(normalized_lemma) + r'\b', re.IGNORECASE)
            for proverb in local_proverb_sentences:
                if pattern.search(proverb):
                    output_data['Surface Word'].append(surface_word)
                    output_data[f'{lang.capitalize()} Lemma'].append(normalized_lemma)
                    output_data['English translation'].append(proverb)
                    output_data['Glossing'].append(affix)
                    output_data['Match Type'].append('Local Proverb Context Match')
                    matched = True
                    break

    df_out = pd.DataFrame(output_data).drop_duplicates(subset=['Surface Word']).reset_index(drop=True)
    df_out.to_csv(f"{lang.lower()}_translated.csv", index=False)
    print(f"Processed {lang}. Output entries: {len(df_out)}")
