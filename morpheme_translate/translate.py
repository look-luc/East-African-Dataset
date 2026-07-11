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

bantuberta_id = "dsfsi/BantuBERTa"
bb_tokenizer = AutoTokenizer.from_pretrained(bantuberta_id)
bb_model = AutoModel.from_pretrained(bantuberta_id).to(device)
bb_model.eval()

script_dir = Path(__file__).resolve().parent.parent
load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

bantu_iso_map = {
    'ganda': 'lug', 'gikuyu': 'kik', 'tshiluba': 'lua',
    'chiga': 'cgg', 'tooro': 'ttj', 'runyoro': 'nyo', 'kamba': 'kam'
}

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
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_bantuberta_embedding(sentence: str, target_word: str):
    inputs = {k: v.to(device) for k, v in bb_tokenizer(sentence, return_tensors="pt").items()}

    word_tokens = bb_tokenizer.tokenize(target_word)
    word_ids = bb_tokenizer.convert_tokens_to_ids(word_tokens)

    if not word_ids:
        return None

    with torch.no_grad():
        outputs = bb_model(**inputs)
        hidden_states = outputs.last_hidden_state.squeeze(0) # Shape: [seq_len, 768]

    input_ids = inputs["input_ids"].squeeze(0).tolist()
    match_indices = []

    for i in range(len(input_ids) - len(word_ids) + 1):
        if input_ids[i:i+len(word_ids)] == word_ids:
            match_indices = list(range(i, i + len(word_ids)))
            break

    if match_indices:
        target_states = hidden_states[match_indices]
        return torch.mean(target_states, dim=0)
    else:
        return torch.mean(hidden_states, dim=0)

def parse_model_outputs(word: str, combined_dict: dict):
    if word in combined_dict:
        lem = combined_dict[word].get('lemmatization', [])
        seg = combined_dict[word].get('segmentation', [])
        predicted_lemma = lem[0] if (isinstance(lem, list) and len(lem) > 0) else word
        affix = seg[0] if (isinstance(seg, list) and len(seg) > 0) else word
        return predicted_lemma, affix
    return None, None

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

def fallback_translation_tier(surface_word: str, lemma: str, lex_exact: dict, lex_sub: dict, panlex_dict: dict):
    """Evaluates lexical priority tiers to locate matching base definitions."""
    norm_surface = strip_accents(surface_word).lower().strip()
    norm_lemma = strip_accents(lemma).lower().strip()

    if norm_surface in lex_exact:
        return lex_exact[norm_surface], 'Lexicon Exact Match'
    if norm_lemma in lex_exact:
        return lex_exact[norm_lemma], 'Lexicon Exact Match'

    if norm_surface in panlex_dict:
        return panlex_dict[norm_surface], 'PanLex Exact Match'
    if norm_lemma in panlex_dict:
        return panlex_dict[norm_lemma], 'PanLex Exact Match'

    if norm_surface in lex_sub:
        return lex_sub[norm_surface], 'Substring Overlap'
    if norm_lemma in lex_sub:
        return lex_sub[norm_lemma], 'Substring Overlap'

    return None, 'No Lexicon Match'

def translation(lang: str, output_name: str, proverbs_file: str):
    lang_folder = script_dir / "data" / lang.lower()

    model_csv = lang_folder / f"{lang.lower().capitalize()}_model_lem_seg.csv"
    if not model_csv.exists():
        print(f"Error: Missing dependency file {model_csv}. Execute model_segment first.")
        return

    df_model = pd.read_csv(model_csv, sep='\t')

    combined_dict = {}
    for _, row in df_model.iterrows():
        w = row['word']
        lem_val = ast.literal_eval(row['lemmatization']) if isinstance(row['lemmatization'], str) else row['lemmatization']
        seg_val = ast.literal_eval(row['segmentation']) if isinstance(row['segmentation'], str) else row['segmentation']
        combined_dict[w] = {'lemmatization': lem_val, 'segmentation': seg_val}

    lex_exact, lex_sub = {}, {}
    dict_path = lang_folder / f"{lang.lower()}_dictionary.csv"
    if dict_path.exists():
        try:
            df_dict = pd.read_csv(dict_path)
            for _, row in df_dict.iterrows():
                k = str(row.get('word', '')).lower().strip()
                v = str(row.get('translation', ''))
                if k and v:
                    lex_exact[k], lex_sub[k] = v, v
        except Exception as e:
            print(f"Lexicon loading skipped for {lang}: {e}")

    iso_code = bantu_iso_map.get(lang.lower(), 'lug')
    panlex_dict = load_panlex(iso_code)

    data_path = script_dir / proverbs_file
    df_data = pd.read_csv(data_path, sep='\t')
    df_lang = df_data[df_data["language"].str.lower() == lang.lower()]

    # List of tuples (Tensor, String) to store verified contextual embeddings safely
    successful_embeddings_pool = []

    print("Building contextual BantuBERTa reference pool maps...")
    for _, row in df_lang.iterrows():
        proverb_text = row['african_proverb']
        clean_proverb = re.sub(r"[^\w\s]", "", proverb_text).strip()
        words_in_proverb = clean_proverb.split()

        for surface_word in words_in_proverb:
            predicted_lemma, _ = parse_model_outputs(surface_word, combined_dict)
            if not predicted_lemma:
                predicted_lemma = surface_word

            translation_string, match_type = fallback_translation_tier(
                surface_word, predicted_lemma, lex_exact, lex_sub, panlex_dict
            )

            if translation_string and match_type != 'No Lexicon Match':
                try:
                    known_vec = get_bantuberta_embedding(clean_proverb, surface_word)
                    if known_vec is not None:
                        successful_embeddings_pool.append((known_vec, translation_string))
                except Exception:
                    continue

    print(f"Contextual Reference Pool Compiled: {len(successful_embeddings_pool)} vectors aligned.")

    output_data = {
        'Surface Word': [],
        f'{lang.capitalize()} Lemma': [],
        'English translation': [],
        'Glossing': [],
        'Match Type': []
    }

    for _, row in df_lang.iterrows():
        proverb_text = row['african_proverb']
        clean_proverb = re.sub(r"[^\w\s]", "", proverb_text).strip()
        words_in_proverb = clean_proverb.split()

        for surface_word in words_in_proverb:
            predicted_lemma, affix = parse_model_outputs(surface_word, combined_dict)
            if not predicted_lemma:
                predicted_lemma, affix = surface_word, surface_word

            translation_string, match_type = fallback_translation_tier(
                surface_word, predicted_lemma, lex_exact, lex_sub, panlex_dict
            )

            matched = False
            if translation_string and match_type != 'No Lexicon Match':
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(predicted_lemma)
                output_data['English translation'].append(translation_string)
                output_data['Glossing'].append(affix)
                output_data['Match Type'].append(match_type)
                matched = True

            # Vector Neighborhood Fallback via Contextual BantuBERTa Tensors
            if not matched and len(successful_embeddings_pool) > 0:
                try:
                    unknown_vec = get_bantuberta_embedding(clean_proverb, surface_word)

                    if unknown_vec is not None:
                        best_score = -1.0
                        best_translation = None

                        for known_vec, known_translation in successful_embeddings_pool:
                            similarity = F.cosine_similarity(
                                unknown_vec.unsqueeze(0),
                                known_vec.unsqueeze(0)
                            ).item()

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
                except Exception:
                    pass

            if not matched:
                output_data['Surface Word'].append(surface_word)
                output_data[f'{lang.capitalize()} Lemma'].append(predicted_lemma)
                output_data['English translation'].append('[Translation Missing]')
                output_data['Glossing'].append(affix)
                output_data['Match Type'].append('No Lexicon Match')

    df_out = pd.DataFrame(output_data).drop_duplicates(subset=['Surface Word']).reset_index(drop=True)
    df_out.to_csv(lang_folder / f"{lang.lower()}_translated.csv", index=False)
    print(f"Processed {lang.upper()} successfully via BantuBERTa pipeline. Output records stored: {len(df_out)}")
