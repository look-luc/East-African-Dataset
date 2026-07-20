import re
import string
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

parent_path = script_dir = Path(__file__).resolve().parent.parent

class translation_final_pass:
    def __init__(
        self,
        model_name:str="dsfsi/BantuBERTa",
        morph_model_name:str="thiomi/bantumorph-v7",
        data_file:str=f"{parent_path}/data/data.csv",
        grammar_file:str=f"{parent_path}/data/bantu_grammar_lookup.csv",
    ) -> None:

        self.data = pd.read_csv(data_file, sep='\t')
        self.grammar_data = pd.read_csv(grammar_file)

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        self.morph_model_name = morph_model_name
        self.morph_tokenizer = AutoTokenizer.from_pretrained(self.morph_model_name)
        self.morph_model = AutoModelForSeq2SeqLM.from_pretrained(self.morph_model_name)

    def _get_embedding(self, text):
        """Generates a dense sequence embedding using contextual mean pooling."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def predict_morphology(self, word: str) -> str:
        inputs = self.morph_tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.morph_model.generate(**inputs, max_new_tokens=64)
        return self.morph_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_noun_class(self, morph_analysis: str) -> str:
        match = re.search(r"(\[N\d+\]|N\d+|BANTU\d+)", morph_analysis, re.IGNORECASE)
        if match:
            raw_tag = match.group(1).upper().strip("[]")
            return re.sub(r"^N(\d+)$", r"BANTU\1", raw_tag)
        return ""

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

        """
        Gets the appropriate files opened for a specific language
        """
        self.lang = lang
        self.translation_keyword = translation_keyword
        self.df_translation = pd.read_csv(
            f"{parent_path}/data/{self.lang.lower()}/{fig_or_lit}_{self.lang.lower()}_random_15 - {fig_or_lit}_{self.lang.lower()}_random_15.csv"
        )
        self.lexicon = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower()}_translated.csv")
        self.lem_seg = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower().capitalize()}_model_lem_seg.csv", sep='\t')

        self.lang_data = pd.DataFrame(self.data[self.data["language"]==self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"]==self.lang])

        """
        gets the gloss that the BantuMorph v7 noun class prediction
        """
        self.glosses = {}
        word_col_lem = 'Surface Word' if 'Surface Word' in self.lem_seg.columns else ('word' if 'word' in self.lem_seg.columns else self.lem_seg.columns[0])

        if hasattr(self, "lem_seg") and "noun class prediction" in self.lem_seg.columns:
            for _, row in self.lem_seg.iterrows():
                w = str(row[word_col_lem])
                raw_pred = str(row["noun class prediction"])
                cleaned_tag = self.extract_noun_class(raw_pred)
                if cleaned_tag:
                    self.glosses[w] = cleaned_tag
        else:
            for word in self.lexicon["Surface Word"].dropna().unique():
                raw_analysis = self.predict_morphology(str(word))
                tag = self.extract_noun_class(raw_analysis)
                if tag:
                    self.glosses[str(word)] = tag

        """
        making a way to get rid of any commas in the proverb
        """
        translator = str.maketrans('', '', string.punctuation)

        """
        goes through each row in the translation file and gets the translation column and finds any '___' of size 2 to 4 and gets the proverb.
        adds the translation as a template and embedds the proverb for later use.
        """
        reference_pool = []
        for r in self.df_translation.itertuples():
            t = getattr(r, self.translation_keyword)
            if isinstance(t, str):
                if re.search(r"_{2,4}(?:\.[\w\d]+)*", t):
                    prov = getattr(r, "african_proverb")
                    if isinstance(prov, str):
                        prov = prov.translate(translator)
                    reference_pool.append({"template": t, "embedding": self._get_embedding(prov)})

        """
        the start of the final pass
        """
        ranked_indices = []
        for row in self.df_translation.itertuples():
            translation = getattr(row, self.translation_keyword)
            is_borrowed = False

            if not isinstance(translation, str):
                """
                getting the embeddings of the current proverb
                """
                current_prov = getattr(row, "african_proverb")
                current_emb = self._get_embedding(current_prov)

                best_template = None
                best_sim = -1.0

                """
                calculating the similarity score for the embeddings
                """
                for ref in reference_pool:
                    similarity = torch.mm(current_emb, ref["embedding"].T).item()
                    # checking if the similarity is better
                    if similarity > best_sim:
                        best_sim = similarity
                        best_template = ref["template"]

                if best_template:
                    translation = best_template
                    is_borrowed = True
                else:
                    ranked_indices.append(translation)
                    continue

            slots = self.extract_slots(translation) # finding all of the instances of the underscored portions
            working_sentence = translation

            if is_borrowed:
                print(f"\n[BantuBERTa Retrieval] Borrowed template for: {getattr(row, 'african_proverb')}")
                print(f"Borrowed Frame: {translation}")

            """
            getting the glossing and surface columns
            """
            gloss_col = next((col for col in self.lexicon.columns if 'gloss' in col.lower()), 'Glossing')
            word_col = 'Surface Word' if 'Surface Word' in self.lexicon.columns else 'word'

            for slot_tag in slots:
                clean_tag = re.sub(r"^_{2,4}", "", slot_tag) # getting rid of the underscores
                tag_components = [
                    re.sub(r"^N(\d+)$", r"BANTU\1", c.upper()) # replacing the noun class gloss from the manual glossing
                    for c in clean_tag.split('.') if c # getting rid of the periods from BantuMorph v7
                ]

                if tag_components:
                    mask = pd.Series(True, index=self.lexicon.index)

                    """
                    making sure that the gloss is in the column given the masked sentence
                    """
                    for comp in tag_components:
                        mask &= self.lexicon[gloss_col].str.contains(comp, na=False, regex=False)
                    candidate_pool = self.lexicon[mask][word_col].dropna().unique().tolist()
                else:
                    candidate_pool = self.lexicon[word_col].dropna().unique().tolist()
                    if tag_components and candidate_pool:
                        filtered_pool = [
                            word for word in candidate_pool
                            if word in self.glosses and any(comp in self.glosses[word] for comp in tag_components)
                        ]
                        candidate_pool = filtered_pool if filtered_pool else candidate_pool

                    chosen_token = self._find_best_candidate(working_sentence, slot_tag, candidate_pool)

                chosen_token = self._find_best_candidate(working_sentence, slot_tag, candidate_pool) # finds the best candidate from the embedding

                if chosen_token: # when found the right token, replaces it in the working sentence
                    match = self.lexicon[self.lexicon[word_col] == chosen_token]['English translation']
                    english_val = match.iloc[0] if not match.empty else chosen_token

                    if not isinstance(english_val, str) or '[Translation Missing]' in english_val:
                        english_val = chosen_token

                    working_sentence = working_sentence.replace(slot_tag, str(english_val), 1)

            """
            going through the rest of the lingering  slots that need to be replaced
            """
            residual_slots = self.extract_slots(working_sentence)
            residual_slots = self.extract_slots(working_sentence)
            if residual_slots:
                global_fallback_pool = self.lexicon[word_col].dropna().unique().tolist()
                for residual_tag in residual_slots:
                    fallback_token = self._find_best_candidate(working_sentence, residual_tag, global_fallback_pool)
                    if fallback_token:
                        match = self.lexicon[self.lexicon[word_col] == fallback_token]['English translation']
                        english_val = match.iloc[0] if not match.empty else fallback_token
                        if not isinstance(english_val, str) or '[Translation Missing]' in english_val:
                            english_val = fallback_token
                        working_sentence = working_sentence.replace(residual_tag, str(english_val), 1)
                    else:
                        working_sentence = working_sentence.replace(residual_tag, "", 1)
            ranked_indices.append(working_sentence)
        return ranked_indices
