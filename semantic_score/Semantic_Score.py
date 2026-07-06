import Levenshtein
import torch
from nltk.translate.chrf_score import sentence_chrf
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_score(model_text:str, gloss_translation:str):
    tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-bert-base')
    model = AutoModel.from_pretrained('efederici/sentence-bert-base')

    encoded_input = tokenizer([model_text, gloss_translation], padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return sentence_embeddings

def compute_chrf(manual_literals: str, model_literals: str):

def compute_structural_metrics(manual_literals: str, model_literals: str):
    results = []

    for idx, (manual, model) in enumerate(zip(manual_literals, model_literals)):
        # ChrF expects a list of reference tokens/sentences, and a candidate string
        # By default, sentence_chrf handles character n-grams (usually up to 6-grams)
        chrf_score = sentence_chrf([manual], model)

        # Absolute Levenshtein Distance (number of edits)
        abs_lev = Levenshtein.distance(manual, model)

        # Normalized Distance (0.0 to 1.0, where 1.0 means completely identical strings)
        max_len = max(len(manual), len(model))
        normalized_similarity = 1.0 - (abs_lev / max_len) if max_len > 0 else 1.0

        results.append({
            "Pair": idx + 1,
            "Manual": manual,
            "Model": model,
            "ChrF": round(chrf_score, 4),
            "Levenshtein_Distance": abs_lev,
            "Levenshtein_Normalized_Sim": round(normalized_similarity, 4)
        })

    return results
