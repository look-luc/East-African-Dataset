import os
import re
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

model_id = "thiomi/bantumorph-v7"

tokenizer = AutoTokenizer.from_pretrained(model_id, token = TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token = TOKEN)

script_dir = Path(__file__).resolve().parent.parent

def query_bantumorph(word: str, tasks: list[str] | None = None):
    """
    Tasks available: 'segmentation', 'lemmatization', 'noun class prediction'
    """
    if tasks is None:
        tasks = ["lemmatization", 'segmentation']

    output = {word: {task: [] for task in tasks}}

    for task in tasks:
        input_text = f"{task}: {word}"
        inputs = tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)

        output[word][task].append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return output

def model_extract(data_title:str, lang:str):
    df = pd.read_csv(str(script_dir / data_path))

    lang_data_df = df[df["language"].lower() == lang.lower()]

    proverbs = lang_data_df["african_proverb"].tolist()

    model_out = []
    for proverb in proverbs:
        clean_proverb = re.sub(r"[^\w\s]", "", proverb).strip()

        for word in clean_proverb:
            model_out.append(query_bantumorph(str(word)))

    combined_dict = {k: v for d in model_out for k, v in d.items()}
    out_df = pd.DataFrame.from_dict(combined_dict, orient='index')

    out_df.to_csv(f"{lang}_model_lem_seg.csv", sep='\t')
