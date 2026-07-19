import re
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
        return re.findall(r"_{2,4}(?:\.[\w\d]+)+", template_sentence)

    def _find_best_candidate(self, native_words: list, target_word_idx: int, candidate_pool: list) -> str:
            """Masks a targeted native word slot and returns the highest scoring candidate from the pool."""
            if not candidate_pool:
                return ""

            best_candidate = candidate_pool[0]
            highest_score = -torch.inf

            for candidate in candidate_pool:
                candidate_tokens = self.tokenizer.tokenize(candidate)

                candidate_ids = self.tokenizer.convert_tokens_to_ids(candidate_tokens)
                num_masks_needed = len(candidate_tokens)

                if all(item == self.tokenizer.unk_token_id for item  in candidate_ids):
                    continue

                masked_words = native_words.copy() # getting the copy of the word to translate
                mask_string = " ".join([self.tokenizer.mask_token] * num_masks_needed)
                masked_words[target_word_idx] = mask_string
                masked_sentence = " ".join(masked_words)

                inputs = self.tokenizer(masked_sentence, return_tensors="pt") # tokenized the whole sentence with the replaced masked token

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    mask_logits = outputs.logits

                mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1] # finding where the masked token is in the inputs

                # fallback where if there isn't anything will be [UNK]
                if len(mask_token_index) != num_masks_needed:
                    continue

                current_candidate_score = 0.0

                for i in range(num_masks_needed):
                    mask_position = mask_token_index[i]
                    target_subword_id = candidate_ids[i]

                    token_logits = mask_logits[0, mask_position, :]

                    log_probabilities = F.log_softmax(token_logits, dim=-1)
                    current_candidate_score = current_candidate_score + log_probabilities[target_subword_id].item()

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
        self.lang_data = pd.DataFrame(self.data[self.data["language"]==self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"]==self.lang])

        reference_pool = []
        for r in self.df_translation.itertuples():
            t = getattr(r, self.translation_keyword)
            if isinstance(t, str):
                # Ensure the reference has valid slot markers before counting it
                if re.search(r"_{2,4}(?:\.[\w\d]+)+", t):
                    prov = getattr(r, "african_proverb")
                    reference_pool.append({"template": t, "embedding": self._get_embedding(prov)})

        ranked_indices = []
        for row in self.df_translation.itertuples():
            translation = getattr(row, self.translation_keyword)
            native_proverb_tokens = getattr(row, "african_proverb").split() # Keep as list for .copy() downstream

            is_borrowed = False

            # If template is missing, execute BantuBERTa similarity retrieval
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

            # Now the slot loop will evaluate perfectly for both original and borrowed frames
            for idx, slot_tag in enumerate(slots):
                clean_tag = re.sub(r"^_{2,4}", "", slot_tag)
                tag_components = [
                    re.sub(r"^N(\d+)$", r"BANTU\1", c.upper())
                    for c in clean_tag.split('.') if c
                ]
                components = ";".join(tag_components)

                gloss_col = 'Glossing' if 'Glossing' in self.lexicon.columns else 'proposed_leipzig_gloss'
                word_col = 'Surface Word' if 'Surface Word' in self.lexicon.columns else 'word'

                candidate_pool = self.lexicon[self.lexicon[gloss_col].str.contains(components)][word_col].dropna().unique().tolist()

                target_word_idx = None
                morpheme_source = getattr(row, "morpheme_breaks", "")
                morpheme_tokens = morpheme_source.split() if isinstance(morpheme_source, str) else []
                for word_idx, native_token in enumerate(morpheme_tokens):
                    native_token_upper = native_token.upper()
                    if all(comp in native_token_upper for comp in tag_components):
                        target_word_idx = word_idx
                        break

                if target_word_idx is None:
                    continue

                chosen_token = self._find_best_candidate(native_proverb_tokens, target_word_idx, candidate_pool) # getting best word to replace

                if chosen_token:
                    working_sentence = working_sentence.replace(slot_tag, str(chosen_token), 1)
            ranked_indices.append(working_sentence)
        return ranked_indices
