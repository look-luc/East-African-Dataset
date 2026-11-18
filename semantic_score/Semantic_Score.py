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

TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
MODEL = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semantic_score(text1: str, text2: str) -> float:
    encoded_input = TOKENIZER([text1, text2], padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = MODEL(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    similarity = F.cosine_similarity(sentence_embeddings[0:1], sentence_embeddings[1:2])
    return similarity.item()

def compute_chrf(manual_literals, model_literals: str):
    chrf_score = [round(sentence_chrf([sent], model_literals), 4) for sent in manual_literals]
    return chrf_score

def levenshtein_score(manual_literals, model_literals):
    abs_levs = []
    normal = []
    for sentence in manual_literals:
        abs_lev = Levenshtein.distance(sentence, model_literals)
        abs_levs.append(abs_lev)
        max_len = max(len(sentence), len(model_literals))
        normal.append(1.0 - (abs_lev / max_len) if max_len > 0 else 1.0)

    return abs_levs, normal

def correct_glossing(corr_gloss: DataFrame, correction_columns: list) -> DataFrame:
    translator = str.maketrans('', '', string.punctuation)
    corr_gloss["combined"] = corr_gloss.astype(str).agg(" ".join, axis=1).str.strip().str.translate(translator).split()
    return corr_gloss.drop(columns=correction_columns)

def map_gloss_prov(df_merged):
    combined_data = {}
    for _, row in df_merged.iterrows():
        proverb = row['african_proverb']
        combined_data[proverb] = {
            "corrected": row['combined'],
            "manual": row['final pass'],
            "model": row['predict']
        }
    return combined_data

def compute_structural_metrics(lang: str, df, corrected_column: str):
    # Construct a proper pandas DataFrame rather than a dictionary
    lang_completed = pd.DataFrame({
        'african_proverb': df['african_proverb'] if 'african_proverb' in df.columns else df.index,
        'combined': df[corrected_column],
        'final pass': df['final pass'] if 'final pass' in df.columns else df[corrected_column],
        'predict': df['predict']
    })

    combined_data = map_gloss_prov(lang_completed)

    results = {}
    for proverb, data in combined_data.items():
        lev_m, norm_m = levenshtein_score(str(data["corrected"]), str(data["model"]))
        lev_man, norm_man = levenshtein_score(str(data["corrected"]), str(data["manual"]))

        results[proverb] = {
            "Levenshtein Distance": {
                "model vs corrected": lev_m,
                "manual vs corrected": lev_man
            },
            "Levenshtein Normalized Sim": {
                "model vs corrected": norm_m,
                "manual vs corrected": norm_man
            },
            "ChrF": {
                "model vs corrected": compute_chrf([str(data["corrected"])], str(data["model"])),
                "manual vs corrected": compute_chrf([str(data["corrected"])], str(data["manual"]))
            },
            "Sentence Bert Embeddings": {
                "model vs corrected": semantic_score(str(data["corrected"]), str(data["model"])),
                "manual vs corrected": semantic_score(str(data["corrected"]), str(data["manual"]))
            }
        }

    results_df = pd.DataFrame.from_dict({(i, j): results[i][j]
                                         for i in results.keys()
                                         for j in results[i].keys()},
                                        orient='index')
    # Convert to LaTeX string (with clean 4-decimal float formatting)
    latex_output = results_df.to_latex(float_format="%.4f")

    # 1. Print directly to console / Slurm log
    print(f"\n--- LaTeX Table for {lang} ---\n{latex_output}")

    # 2. Save to a .tex file
    output_dir = script_path / "results" / "latex"
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_file = output_dir / f"{lang.lower()}_metrics.tex"
    tex_file.write_text(latex_output, encoding="utf-8")

    print(f"Saved LaTeX table to: {tex_file}")
