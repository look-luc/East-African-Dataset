import re
import string
from itertools import batched
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer

parent_path = script_dir = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path('.').resolve()

class translation_final_pass:
    def __init__(
        self,
        model_name: str = "dsfsi/BantuBERTa",
        morph_model_name: str = "thiomi/bantumorph-v7",
        data_file: str = f"{parent_path}/data/data.csv",
        grammar_file: str = f"{parent_path}/data/bantu_grammar_lookup.csv",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = pd.read_csv(data_file, sep='\t')
        self.grammar_data = pd.read_csv(grammar_file)

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)

        self.morph_model_name = morph_model_name
        self.morph_tokenizer = AutoTokenizer.from_pretrained(self.morph_model_name)
        self.morph_model = AutoModelForSeq2SeqLM.from_pretrained(self.morph_model_name).to(self.device)
        self.morph_cache: dict[str, str] = {}

    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return F.normalize(embeddings, p=2, dim=1)

    def predict_morphology(self, words: str | list[str], batch_size:int=64)-> str | dict[str, str]:
        self.batch_size = batch_size
        single = isinstance(words, str)

        word_list = [words] if single else [str(w) for w in words]
        unvisited_words = [w for w in word_list if w not in self.morph_cache]

        if unvisited_words:
            for batch in batched(unvisited_words, batch_size):
                inputs = self.morph_tokenizer(list(batch), padding=True, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    output_tokens = self.morph_model.generate(**inputs, max_new_tokens=64)
                decoded_analyses = self.morph_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

                for word_item, analysis in zip(batch, decoded_analyses):
                    self.morph_cache[word_item] = analysis

        if single:
            return self.morph_cache.get(words, "")
        return {w: self.morph_cache[w] for w in word_list if w in self.morph_cache}

    def extract_morph_tags(self, morph_analysis: str) -> set[str]:
        if not morph_analysis:
            return set()

        bracket_matches = re.findall(r"\[(.*?)\]", morph_analysis)
        tags = set()

        if bracket_matches:
            for match in bracket_matches:
                for sub in match.split('.'):
                    clean = sub.strip().upper()
                    if clean:
                        clean = re.sub(r"^N(\d+)$", r"BANTU\1", clean)
                        tags.add(clean)
        else:
            for tok in re.split(r"[\s\-\_]+", morph_analysis):
                clean = tok.strip("[]").upper()
                if clean:
                    clean = re.sub(r"^N(\d+)$", r"BANTU\1", clean)
                    tags.add(clean)

        return tags

    def extract_slots(self, template_sentence: str) -> list[str]:
        return re.findall(r"_{2,4}(?:\.[\w\d]+)*", template_sentence)

    def _find_best_candidate(self, working_sentence: str, slot_tag: str, candidate_pool: list, batch_size:int=64) -> str:
        self.batch_size = batch_size
        if not candidate_pool:
            return ""

        candidate_pool_str = [str(item) for item in candidate_pool]
        candidate_tokenize = self.tokenizer(candidate_pool_str, add_special_tokens=False)

        best_candidate = str(candidate_pool_str[0])
        highest_score = -float('inf')

        candidate_groups = {}

        for word, token_ids in zip(candidate_pool_str, candidate_tokenize["input_ids"]):
            length = len(token_ids)
            if length not in candidate_groups:
                candidate_groups[length] = []
            element = (word, token_ids)
            candidate_groups[length].append(element)

        for length, candidates in candidate_groups.items():
            num_masks = max(1, length)
            mask_str = " ".join([self.tokenizer.mask_token] * num_masks)

            sentence = working_sentence.replace(slot_tag, mask_str, 1)

            inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model(**inputs).logits
                log_probs = F.log_softmax(outputs, dim=-1)

            mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]

            for candidate, candidate_idx in candidates:
                current_candidate_score = 0.0
                for pos_idx in range(length):
                    mask_pos = mask_token_index[pos_idx]
                    token_id = candidate_idx[pos_idx]
                    current_candidate_score += log_probs[0, mask_pos, token_id]
                avg = current_candidate_score / length
                if avg > highest_score:
                    highest_score = current_candidate_score
                    best_candidate = candidate

        return best_candidate

    def _normalize_lexicon(self) -> dict:
        target_col = next(
            (col for col in self.lexicon.columns if 'english' in col.lower() or 'translation' in col.lower()),
            'English translation'
        )
        word_col = 'Surface Word' if 'Surface Word' in self.lexicon.columns else ('word' if 'word' in self.lexicon.columns else self.lexicon.columns[0])

        lookup_map = {}
        for w, t in zip(self.lexicon[word_col], self.lexicon[target_col]):
            clean_w = str(w).strip().lower()
            clean_t = str(t).strip()

            if clean_t and '[Translation Missing]' not in clean_t and clean_t.lower() != 'nan':
                lookup_map[clean_w] = clean_t

        return lookup_map

    def resolve_slot_translation(self, chosen_token: str, slot_tag: str = "") -> str:
        clean_token = str(chosen_token).strip().lower()

        if clean_token in self.lexicon_map:
            return self.lexicon_map[clean_token]

        stripped_token = clean_token.translate(str.maketrans("", "", string.punctuation))
        if stripped_token in self.lexicon_map:
            return self.lexicon_map[stripped_token]

        if hasattr(self, 'lem_map') and clean_token in self.lem_map:
            return self.lem_map[clean_token]

        if hasattr(self, 'morph_model'):
            raw_analysis = self.predict_morphology(clean_token)
            tags = self.extract_morph_tags(raw_analysis)

            root_match = re.search(r"\-\s*([\w]+)$", raw_analysis)
            root = root_match.group(1).lower() if root_match else ""

            translated_root = self.lem_map.get(root, root) if hasattr(self, 'lem_map') else root

            translated_tags = []
            if hasattr(self, 'grammar_map'):
                for t in tags:
                    alt_tag = re.sub(r"^BANTU(\d+)$", r"N\1", t)
                    if t in self.grammar_map:
                        translated_tags.append(self.grammar_map[t])
                    elif alt_tag in self.grammar_map:
                        translated_tags.append(self.grammar_map[alt_tag])

            parts = []
            if translated_tags:
                parts.extend(translated_tags)
            if translated_root:
                parts.append(translated_root)

            if parts:
                return " ".join(parts)

        if slot_tag and hasattr(self, 'grammar_map'):
            clean_tag = re.sub(r"^_{2,4}\.?", "", slot_tag).upper()
            components = [c for c in clean_tag.split('.') if c]
            resolved = []
            for c in components:
                bantu_c = re.sub(r"^N(\d+)$", r"BANTU\1", c)
                norm_c = re.sub(r"^BANTU(\d+)$", r"N\1", c)
                if c in self.grammar_map:
                    resolved.append(self.grammar_map[c])
                elif bantu_c in self.grammar_map:
                    resolved.append(self.grammar_map[bantu_c])
                elif norm_c in self.grammar_map:
                    resolved.append(self.grammar_map[norm_c])
                else:
                    resolved.append(c.lower())
            if resolved:
                return " ".join(resolved)

        return chosen_token

    def ranked_translation(self, fig_or_lit: str, translation_keyword: str = "translation", lang: str | None = None):
        if lang is None:
            raise ValueError("Provide a language")

        self.lang = lang
        self.translation_keyword = translation_keyword
        self.df_translation = pd.read_csv(
            f"{parent_path}/data/{self.lang.lower()}/{fig_or_lit}_{self.lang.lower()}_random_15 - {fig_or_lit}_{self.lang.lower()}_random_15.csv"
        )
        self.lexicon = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower()}_translated.csv")
        self.lem_seg = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower().capitalize()}_model_lem_seg.csv", sep='\t')

        self.lang_data = pd.DataFrame(self.data[self.data["language"] == self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"] == self.lang])

        self.lexicon_map = self._normalize_lexicon()

        word_col_lem = 'Surface Word' if 'Surface Word' in self.lem_seg.columns else ('word' if 'word' in self.lem_seg.columns else self.lem_seg.columns[0])
        target_col_lem = next((col for col in self.lem_seg.columns if 'english' in col.lower() or 'translation' in col.lower()), self.lem_seg.columns[-1])
        self.lem_map = dict(zip(self.lem_seg[word_col_lem].astype(str).str.lower(), self.lem_seg[target_col_lem].astype(str)))

        tag_col_gram = 'tag' if 'tag' in self.grammar_lookup.columns else self.grammar_lookup.columns[0]
        target_col_gram = next(
            (col for col in self.grammar_lookup.columns if 'english' in col.lower() or 'translation' in col.lower()),
            self.grammar_lookup.columns[-1]
        )
        self.grammar_map = dict(zip(self.grammar_lookup[tag_col_gram].astype(str).str.upper(), self.grammar_lookup[target_col_gram].astype(str)))

        self.glosses: dict[str, set[str]] = {}
        word_col_lex = 'Surface Word' if 'Surface Word' in self.lexicon.columns else ('word' if 'word' in self.lexicon.columns else self.lexicon.columns[0])
        unique_lex_words = [str(w) for w in self.lexicon[word_col_lex].dropna().unique()]
        self.predict_morphology(unique_lex_words, batch_size=64)

        for word in unique_lex_words:
            raw_analysis = self.morph_cache.get(word, "")
            tags = self.extract_morph_tags(raw_analysis)
            if tags:
                self.glosses[word] = tags

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

            if not isinstance(translation, str) or pd.isna(translation):
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

                if not isinstance(translation, str) or pd.isna(translation):
                    ranked_indices.append(str(current_prov) if current_prov else "")
                    continue

            slots = self.extract_slots(translation)
            working_sentence = translation

            if is_borrowed:
                print(f"\n[BantuBERTa Retrieval] Borrowed template for: {getattr(row, 'african_proverb')}")
                print(f"Borrowed Frame: {translation}")

            gloss_col = next((col for col in self.lexicon.columns if 'gloss' in col.lower()), 'Glossing')
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
                        if gloss_col in self.lexicon.columns:
                            mask &= self.lexicon[gloss_col].str.contains(comp, na=False, regex=False)
                    candidate_pool = self.lexicon[mask][word_col].dropna().unique().tolist()
                else:
                    candidate_pool = self.lexicon[word_col].dropna().unique().tolist()

                if tag_components and candidate_pool and self.glosses:
                    filtered_pool = [
                        word for word in candidate_pool if str(word) in self.glosses and any(
                            comp in self.glosses[str(word)]
                            or any(comp in tag for tag in self.glosses[str(word)])
                            for comp in tag_components
                        )
                    ]
                    candidate_pool = filtered_pool if filtered_pool else candidate_pool

                chosen_token = self._find_best_candidate(working_sentence, slot_tag, candidate_pool)

                if chosen_token:
                    english_val = self.resolve_slot_translation(chosen_token, slot_tag=slot_tag)
                    working_sentence = working_sentence.replace(slot_tag, str(english_val), 1)

            residual_slots = self.extract_slots(working_sentence)
            if residual_slots:
                global_fallback_pool = self.lexicon[word_col].dropna().unique().tolist()
                for residual_tag in residual_slots:
                    fallback_token = self._find_best_candidate(working_sentence, residual_tag, global_fallback_pool)
                    if fallback_token:
                        english_val = self.resolve_slot_translation(fallback_token, slot_tag=residual_tag)
                        working_sentence = working_sentence.replace(residual_tag, str(english_val), 1)
                    else:
                        working_sentence = working_sentence.replace(residual_tag, "", 1)
            ranked_indices.append(working_sentence)
        return ranked_indices
