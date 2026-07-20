import re
import string
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

    def _get_embedding(self, text):
        """Generates a dense sequence embedding using contextual mean pooling."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def extract_slots(self, template_sentence: str)->list[str]:
        return re.findall(r"_{2,4}(?:\.[\w\d]+)*", template_sentence)

    def _find_best_candidate(self, working_sentence: str, slot_tag: str, candidate_pool: list) -> str:
        if not candidate_pool:
            return ""

        best_candidate = candidate_pool[0]
        highest_score = -torch.inf

        for candidate in candidate_pool:
            candidate_tokens = self.tokenizer.tokenize(candidate)
            candidate_ids = self.tokenizer.convert_tokens_to_ids(candidate_tokens)
            num_masks_needed = len(candidate_tokens)

            if all(item == self.tokenizer.unk_token_id for item in candidate_ids):
                continue

            mask_string = " ".join([self.tokenizer.mask_token] * num_masks_needed)
            masked_sentence = working_sentence.replace(slot_tag, mask_string, 1)

            inputs = self.tokenizer(masked_sentence, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                mask_logits = outputs.logits

            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

            if len(mask_token_index) != num_masks_needed:
                continue

            current_candidate_score = 0.0

            for i in range(num_masks_needed):
                mask_position = mask_token_index[i]
                target_subword_id = candidate_ids[i]

                token_logits = mask_logits[0, mask_position, :]
                log_probabilities = F.log_softmax(token_logits, dim=-1)
                current_candidate_score += log_probabilities[target_subword_id].item()

            if current_candidate_score > highest_score:
                highest_score = current_candidate_score
                best_candidate = candidate

        return best_candidate

    def ranked_translation(self, fig_or_lit:str, translation_keyword:str="translation", lang:str|None=None,):
        if lang is None:
            raise ValueError("Provide a language")

        self.lang = lang
        self.translation_keyword = translation_keyword
        self.df_translation = pd.read_csv(
            f"{parent_path}/data/{self.lang.lower()}/{fig_or_lit}_{self.lang.lower()}_random_15 - {fig_or_lit}_{self.lang.lower()}_random_15.csv"
        )
        self.lexicon = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower()}_translated.csv")
        self.lem_seg = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower().capitalize()}_model_lem_seg.csv", sep='\t')

        self.lang_data = pd.DataFrame(self.data[self.data["language"]==self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"]==self.lang])

        self.glosses = []
        for _, row in self.lem_seg.iterrows():
            g = row["noun class prediction"]
            glossed = g.strip(" ")[1]
            self.glosses.append(glossed[:-1])

        translator = str.maketrans('', '', string.punctuation)

        reference_pool = []
        for r in self.df_translation.itertuples():
            t = getattr(r, self.translation_keyword)
            if isinstance(t, str):
                if re.search(r"_{2,4}(?:\.[\w\d]+)*", t):
                    prov = getattr(r, "african_proverb")
                    if isinstance(prov, str):
                        prov = prov.translate(translator)
                    reference_pool.append({"template": t, "embedding": self._get_embedding(prov)})

        ranked_indices = []
        for row in self.df_translation.itertuples():
            translation = getattr(row, self.translation_keyword)
            is_borrowed = False

            if not isinstance(translation, str):
                current_prov = getattr(row, "african_proverb")
                current_emb = self._get_embedding(current_prov)

                best_template = None
                best_sim = -1.0

                for ref in reference_pool:
                    similarity = torch.mm(current_emb, ref["embedding"].T).item()
                    if similarity > best_sim:
                        best_sim = similarity
                        best_template = ref["template"]

                if best_template:
                    translation = best_template
                    is_borrowed = True
                else:
                    ranked_indices.append(translation)
                    continue

            slots = self.extract_slots(translation)
            working_sentence = translation

            if is_borrowed:
                print(f"\n[BantuBERTa Retrieval] Borrowed template for: {getattr(row, 'african_proverb')}")
                print(f"Borrowed Frame: {translation}")

            gloss_col = next((col for col in self.glosses if 'gloss' in col.lower()), 'gloss')
            word_col = 'Surface Word' if 'Surface Word' in self.lexicon.columns else 'word'

            for slot_tag in slots:
                clean_tag = re.sub(r"^_{2,4}", "", slot_tag)
                tag_components = [
                    re.sub(r"^N(\d+)$", r"BANTU\1", c.upper())
                    for c in clean_tag.split('.') if c
                ]

                if tag_components:
                    mask = pd.Series(True, index=self.lexicon.index)
                    for comp in tag_components:
                        mask &= self.lexicon[gloss_col].str.contains(comp, na=False, regex=False)
                    candidate_pool = self.lexicon[mask][word_col].dropna().unique().tolist()
                else:
                    candidate_pool = self.lexicon[word_col].dropna().unique().tolist()

                chosen_token = self._find_best_candidate(working_sentence, slot_tag, candidate_pool)

                if chosen_token:
                    working_sentence = working_sentence.replace(slot_tag, str(chosen_token), 1)

            residual_slots = self.extract_slots(working_sentence)
            if residual_slots:
                global_fallback_pool = self.lexicon[word_col].dropna().unique().tolist()
                for residual_tag in residual_slots:
                    fallback_token = self._find_best_candidate(working_sentence, residual_tag, global_fallback_pool)
                    if fallback_token:
                        working_sentence = working_sentence.replace(residual_tag, str(fallback_token), 1)
                    else:
                        working_sentence = working_sentence.replace(residual_tag, "", 1)
            ranked_indices.append(working_sentence)
        return ranked_indices
