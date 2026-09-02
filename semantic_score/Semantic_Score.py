import string
from pathlib import Path

import Levenshtein
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.chrf_score import sentence_chrf
from pandas import DataFrame
from transformers import AutoModel, AutoTokenizer

script_path = Path(__file__).resolve().parent.parent

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_score(model_text:str, gloss_translation:str):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    encoded_input = tokenizer([model_text, gloss_translation], padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings

def compute_chrf(manual_literals: str, model_literals: str):
    chrf_score = sentence_chrf([manual_literals], model_literals)
    round(chrf_score, 4)

def levenshtein_score(manual_literals: str, model_literals: str):
    abs_lev = Levenshtein.distance(manual_literals, model_literals)

    max_len = max(len(manual_literals), len(model_literals))
    normalized_similarity = 1.0 - (abs_lev / max_len) if max_len > 0 else 1.0

    return abs_lev, normalized_similarity

def correct_glossing(corr_gloss:DataFrame, correction_columns:list):
    translator = str.maketrans('', '', string.punctuation)

    corr_gloss["combined"] = corr_gloss.astype(str).agg("\n", axis=1).strip().translate(translator)
    return corr_gloss.drop(columns=correction_columns)

def map_gloss_prov(corrected, manual, model, proverbs):
    combined_data = {}
    for proverb in proverbs:
        combined_data[proverb]["corrected"] = corrected.loc[corrected['proverb'] == proverb, 'combined'].item()
        combined_data[proverb]["manual"] = manual.loc[manual['proverb'] == proverb, 'final pass'].item()
        combined_data[proverb]["model"] = manual.loc[model['proverb'] == proverb, 'predict'].item()
    return combined_data

def compute_structural_metrics(lang:str, gloss_rel_path:str):
    lang_completed = pd.read_csv(f"{script_path}/{gloss_rel_path}")

    manual_gloss = lang_completed["final pass"].to_frame()
    african_proverb = lang_completed["african_proverb"].to_frame()
    correct_gloss = lang_completed[["african_proverb", "Correction", "Correction 2", "Correction 3"]].to_frame()
    correct_gloss = correct_glossing(correct_gloss, ["Correction", "Correction 2", "Correction 3"])
    model_gloss = pd.read_csv(f"{script_path}/data/data.csv")["predict"]

    combined_data = map_gloss_prov(correct_gloss, manual_gloss, model_gloss, african_proverb)

    data_df = pd.read_csv(f"{script_path}/data/data.csv")
    results = {}
    for proverb in combined_data.keys():
        levenshtein_model, normalized_model = levenshtein_score(combined_data[proverb]["corrected"], combined_data[proverb]["model"])
        levenshtein_manual, normalized_manual = levenshtein_score(combined_data[proverb]["corrected"], combined_data[proverb]["manual"])

        results[proverb]["Levenshtein Distance"]["model vs corrected"] = levenshtein_model
        results[proverb]["Levenshtein Normalized Sim"]["model vs corrected"] = normalized_model

        results[proverb]["Levenshtein Distance"]["manual vs corrected"] = levenshtein_manual
        results[proverb]["Levenshtein Normalized Sim"]["manual vs corrected"] = normalized_manual

        results[proverb]["ChrF"]["model vs corrected"] = compute_chrf(combined_data[proverb]["corrected"], combined_data[proverb]["model"])
        results[proverb]["ChrF"]["manual vs corrected"] = compute_chrf(combined_data[proverb]["corrected"], combined_data[proverb]["manual"])

        results[proverb]["Sentence Bert Embeddings"]["model vs corrected"] = semantic_score(
            combined_data[proverb]["corrected"], combined_data[proverb]["model"]
        )
        results[proverb]["Sentence Bert Embeddings"]["manual vs corrected"] = semantic_score(
            combined_data[proverb]["corrected"], combined_data[proverb]["manual"]
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{script_path}/{lang}_results.csv", sep='\t')
