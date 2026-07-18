import re
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

    def _get_embedding(self, text):
        """Generates a dense sequence embedding using contextual mean pooling."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def extract_slots(self, template_sentence: str)->list[str]:
        return re.findall(r"__(?_)(?_)(?:\.[\w\d]+)+", template_sentence)

    def ranked_translation(self, fig_or_lit:str, translation_keyword:str="translation"):
        self.translation_keyword = translation_keyword
        self.df_translation = pd.read_csv(f"{parent_path}/data/{fig_or_lit}_{self.lang.lower()}_random_15.csv")
        self.lang_data = pd.DataFrame(self.data[self.data["language"]==self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"]==self.lang])

        ranked_indices = []

        for row in self.df_translation.itertuples():
            translation = getattr(row, self.translation_keyword)

            slots = self.extract_slots(translation)

            if not slots:
                return translation

            current_context = f"[Language: {self.lang.lower()}] Sentence Structure: "
            working_sentence = translation

            print(f"\nProcessing Sentence Frame for Language: [{self.lang.upper()}]")
            print(f"Original Input: {translation}")
            print("-" * 50)

            for idx, slot_tag in enumerate(slots):
                clean_tag = slot_tag.replace(r"__?_?_.", "")

                gloss_col = 'Glossing' if 'Glossing' in self.grammar_data.columns else 'proposed_leipzig_gloss'
                word_col = 'Surface Word' if 'Surface Word' in self.grammar_data.columns else 'word'
