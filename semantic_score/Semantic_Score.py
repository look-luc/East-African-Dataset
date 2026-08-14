from pathlib import Path

import Levenshtein
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.translate.chrf_score import sentence_chrf
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

def compute_structural_metrics(lang:str, gloss_rel_path:str, translation_column:str=""):
    manual_gloss_df = pd.read_csv(f"{script_path}/{gloss_rel_path}")
    manual_gloss_df = manual_gloss_df[["african_proverb", translation_column]]
    model_gloss = pd.read_csv(f"{script_path}/data/data.csv")["predict"]
    data_df = pd.read_csv(f"{script_path}/data/data.csv")
    results = {}
    for idx, row in manual_gloss_df.iterrows():
        proverb = row["african_proverb"]
        matched_rows = data_df[data_df["african_proverbs"] == proverb]
        if not matched_rows.empty:
            config_name = matched_rows["experiment_config"].iloc[0]
            results[f"{config_name} ({lang.lower().capitalize()})"] = {
                "Sentence Bert Embeddings": [],
                "ChrF": [],
                "Levenshtein Distance": [],
                "Levenshtein Normalized_Sim": [],
                "Proverb": proverb
            }
        manual_literals = matched_rows[[translation_column]]

        for idx, (manual, model) in enumerate(zip(manual_literals, model_gloss)):
            levenshtein, normalized = levenshtein_score(manual, model)

            results[lang]["Sentence Bert Embeddings"].append(semantic_score(manual, model))
            results[lang]["ChrF"].append(compute_chrf(manual, model))
            results[lang]["Levenshtein Distance"].append(levenshtein)
            results[lang]["Levenshtein Normalized_Sim"].append(normalized)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{script_path}/{lang}_results.csv", sep='\t')
