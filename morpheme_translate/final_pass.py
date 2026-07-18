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
    ) -> None:

        self.data = pd.read_csv(data_file, sep='\t')
        self.grammar_data = pd.read_csv(grammar_file)

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
        return re.findall(r"_{2,4}(?:\.[\w\d]+)+", template_sentence)

    def ranked_translation(self, fig_or_lit:str, translation_keyword:str="translation", lang:str|None=None,):
        if lang is None:
            raise ValueError("Provide a language")

        self.lang = lang

        self.translation_keyword = translation_keyword
        self.df_translation = pd.read_csv(
            f"{parent_path}/data/{self.lang}/{fig_or_lit}_{self.lang.lower()}_random_15-{fig_or_lit}_{self.lang}_random_15.csv"
        )
        self.lexicon = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}_translated.csv")
        self.lang_data = pd.DataFrame(self.data[self.data["language"]==self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"]==self.lang])

        ranked_indices = []

        for row in self.df_translation.itertuples():
            translation = getattr(row, self.translation_keyword)

            if not isinstance(translation, str):
                ranked_indices.append(translation)
                continue

            slots = self.extract_slots(translation)

            if not slots:
                ranked_indices.append(translation)
                continue

            current_context = f"[Language: {self.lang.lower()}] Sentence Structure: "
            working_sentence = translation

            print(f"\nProcessing Sentence Frame for Language: [{self.lang.upper()}]")
            print(f"Original Input: {translation}")
            print("-" * 50)

            for idx, slot_tag in enumerate(slots):
                clean_tag = re.sub(r"^_{2,4}\.", "", slot_tag)

                gloss_col = 'Glossing' if 'Glossing' in self.lexicon.columns else 'proposed_leipzig_gloss'
                word_col = 'Surface Word' if 'Surface Word' in self.lexicon.columns else 'word'

                candidate_pool = self.lexicon[self.lexicon[gloss_col] == clean_tag][word_col].dropna().unique().tolist()

                if not candidate_pool:
                    candidate_pool = self.grammar_lookup[
                        self.grammar_lookup['proposed_leipzig_gloss'] == clean_tag
                    ]['morpheme_segment'].dropna().unique().tolist()

                if not candidate_pool:
                    print(f"⚠️ No vocabulary matches found for tag '{clean_tag}'. Skipping slot.")
                    continue

                prompt = f"{current_context} Frame: {working_sentence.replace(slot_tag, '[MASK]', 1)}"
                target_vector = self._get_embedding(prompt)

                candidate_vectors = torch.cat([self._get_embedding(str(c)) for c in candidate_pool], dim=0)
                scores = torch.matmul(target_vector, candidate_vectors.T).squeeze(0)

                best_idx = torch.argmax(scores).item()
                chosen_token = str(candidate_pool[best_idx])

                working_sentence = working_sentence.replace(slot_tag, chosen_token, 1)
                current_context += f" {chosen_token}"

                print(f"Filled Slot {idx + 1} ({slot_tag}) ➔ '{chosen_token}' (Confidence: {scores[best_idx].item():.4f})")

            ranked_indices.append(working_sentence)
        return ranked_indices
