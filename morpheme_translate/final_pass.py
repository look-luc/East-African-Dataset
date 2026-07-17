from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

parent_path = script_dir = Path(__file__).resolve().parent.parent

class translation_final_pass:
    def __init__(
        self,
        model_name:str="dsfsi/BantuBERTa",
        data_file:str=f"{parent_path}/data/data.csv",
        grammar_file:str=f"{parent_path}/data/bantu_grammar_lookup.csv",
        lang:str|None=None,
    ) -> None:
        if lang is None:
            raise ValueError("Provide a language")

        self.data = pd.read_csv(data_file, sep='\t')
        self.grammar_data = pd.read_csv(grammar_file)

        self.lang = lang

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, text):
        """Generates a dense sequence embedding using contextual mean pooling."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Average across the sequence length dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def ranked_translation(self, fig_or_lit:str):
        self.df_translation = pd.read_csv(f"{parent_path}/data/{fig_or_lit}_{self.lang.lower()}_random_15.csv")
        self.lang_data = pd.DataFrame(self.data[self.data["language"]==self.lang])

        target_context = f"[Language: {self.lang.capitalize()}] Verb Root: okusoma (.nfin). Inflected form is: "
