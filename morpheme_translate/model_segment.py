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
    inputs = {k: v.to(device) for k, v in tokenizer(word, return_tensors="pt").items()}

    with torch.no_grad():
        outputs = encoder(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        mean_pooled = torch.mean(hidden_states, dim=1).squeeze(0)
    return mean_pooled


def query_bantumorph(words: list[str], batch_size: int = 64):
    """Processes multiple words and tasks in parallel batches on the GPU."""
    tasks = ["lemmatization", "segmentation", "noun class prediction"]

    prompts = []
    prompt_metadata = []

    for word in words:
        for task in tasks:
            prompts.append(f"{task}: {word}")
            prompt_metadata.append((word, task))

    results = {word: {task: [] for task in tasks} for word in words}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_meta = prompt_metadata[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for (word, task), decoded_text in zip(batch_meta, decoded_outputs):
            results[word][task].append(decoded_text)

    return results

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
        words_in_proverb = clean_proverb.split()
        batch_results = query_bantumorph(words_in_proverb)

        for word in words_in_proverb:
            if word not in seen_words:
                seen_words.add(word)

                word_data: dict = batch_results[word]

                word_data['embedding'] = get_embeddings(word)

                model_out.append({word: word_data})

    combined_dict = {k: v for d in model_out for k, v in d.items()}
    out_df = pd.DataFrame.from_dict(combined_dict, orient='index')

    out_df.index.name = 'surface_word'

    out_df.to_csv(lang_folder / f"{lang.lower().capitalize()}_model_lem_seg.csv", sep='\t', index=True)
