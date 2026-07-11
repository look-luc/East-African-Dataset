import os
import re
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

model_id = "thiomi/bantumorph-v7"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=TOKEN)
model = model.to(device)

script_dir = Path(__file__).resolve().parent.parent

encoder = model.get_encoder()
encoder.eval()

def get_embeddings(word: str):
    inputs = tokenizer(word, return_tensors="pt")

    with torch.no_grad():
        outputs = encoder(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        mean_pooled = torch.mean(hidden_states, dim=1).squeeze(0)
    return mean_pooled.tolist()


def query_bantumorph(word: str, tasks: list[str] | None = None):
    if tasks is None:
        tasks = ["lemmatization", "segmentation", "noun class prediction"]

    output = {word: {task: [] for task in tasks}}
    for task in tasks:
        input_text = f"{task}: {word}"
        inputs = {k: v.to(device) for k, v in tokenizer(input_text, return_tengers="pt").items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)
        output[word][task].append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return output

def model_extract(data_title: str, lang: str):
    lang_folder = script_dir / "data" / lang.lower()
    lang_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(script_dir / data_title, sep='\t')

    lang_data_df = df[df["language"].str.lower() == lang.lower()]

    proverbs = lang_data_df["african_proverb"].tolist()

    model_out = []
    seen_words = set()

    for proverb in proverbs:
        clean_proverb = re.sub(r"[^\w\s]", "", proverb).strip()
        for word in clean_proverb.split():
            if word not in seen_words:
                seen_words.add(word)

                res = query_bantumorph(word)
                res[word]['embedding'] = get_embeddings(word)
                model_out.append(res)

    combined_dict = {k: v.to(device) for d in model_out for k, v in d.items()}
    out_df = pd.DataFrame.from_dict(combined_dict, orient='index')

    out_df.index.name = 'surface_word'

    out_df.to_csv(lang_folder / f"{lang.lower().capitalize()}_model_lem_seg.csv", sep='\t', index=True)
