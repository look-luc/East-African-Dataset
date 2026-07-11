import os
import re
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

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

def batch_get_embeddings(words: list[str], batch_size: int = 128) -> dict[str, torch.Tensor]:
    """Generates embeddings for all unique words at once using large parallel GPU matrices."""
    word_to_embedding = {}
    if not words:
        return word_to_embedding

    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        inputs = tokenizer(batch_words, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = encoder(**inputs)

            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs

            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            sum_embeddings = torch.sum(masked_hidden, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        for word, emb in zip(batch_words, mean_pooled):
            word_to_embedding[word] = emb

    return word_to_embedding


def query_bantumorph(words: list[str], batch_size: int = 128) -> dict[str, dict]:
    """Generates task annotations in parallel batches on the GPU."""
    tasks = ["lemmatization", "segmentation", "noun class prediction"]
    prompts = []
    prompt_metadata = []

    for word in words:
        for task in tasks:
            prompts.append(f"{task}: {word}")
            prompt_metadata.append((word, task))

    results = {word: {task: [] for task in tasks} for word in words}

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_meta = prompt_metadata[i:i + batch_size]

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

    all_unique_words = set()
    for proverb in proverbs:
        clean_proverb = re.sub(r"[^\w\s]", "", proverb).strip()
        all_unique_words.update(clean_proverb.split())

    unique_words_list = list(all_unique_words)
    print(f"Total unique words to process for {lang}: {len(unique_words_list)}")

    print("Running parallel task generation batches...")
    global_batch_results = query_bantumorph(unique_words_list, batch_size=128)

    print("Running parallel encoder embedding batches...")
    global_embeddings = batch_get_embeddings(unique_words_list, batch_size=128)

    combined_dict = {}
    for word in unique_words_list:
        word_data: dict = global_batch_results[word]
        word_data['embedding'] = global_embeddings[word]
        combined_dict[word] = word_data

    final_records = []
    for word, data in combined_dict.items():
        final_records.append({
            "word": word,
            "lemmatization": data["lemmatization"],
            "segmentation": data["segmentation"],
            "noun class prediction": data["noun class prediction"],
            "embedding": data["embedding"].cpu().tolist()
        })

    out_df = pd.DataFrame(final_records)
    out_df.to_csv(lang_folder / f"{lang.lower().capitalize()}_model_lem_seg.csv", sep='\t', index=False)
    print(f"Completed processing {lang}. Extraction output saved.")
