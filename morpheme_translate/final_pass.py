import re
import string
from itertools import batched
from pathlib import Path

import nltk
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.corpus import words
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

parent_path = script_dir = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path('.').resolve()
nltk.download('words')

class translation_final_pass:
    def __init__(
        self,
        model_name: str = "dsfsi/BantuBERTa",
        morph_model_name: str = "thiomi/bantumorph-v7",
        translation_model_name:str = "facebook/nllb-200-distilled-600M",
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

        self.translation_model_name = translation_model_name

        self.batch_size = 64

    def _get_embedding(self, text):
        all_embeddings = []
        if isinstance(text, str):
            text = [text]

        for batch in batched(text, self.batch_size):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

                # Mask out padding tokens during mean calculation
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)

                mean_pooled = sum_embeddings / sum_mask
                normalize = F.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(normalize)
        return torch.cat(all_embeddings, dim=0)

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

    def _find_best_candidate(self, working_sentence: str, slot_tag: str, candidate_pool: list, batch_size: int = 64) -> str:
        self.batch_size = batch_size

        best_candidate = ""
        highest_score = -float('inf')

        length_groups = {}

        for candidate in candidate_pool:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            length = len(token_ids)
            if length == 0:
                continue
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((candidate, token_ids))

        for length, candidates in length_groups.items():
            if length == 0:
                continue
            mask_str = " ".join([self.tokenizer.mask_token] * length)
            masked_sentence = working_sentence.replace(slot_tag, mask_str, 1)
            input = self.tokenizer(masked_sentence, truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                outputs = self.model(**input).logits
                log_probs = F.log_softmax(outputs, dim=-1)

            mask_pos = torch.where(input["input_ids"] == self.tokenizer.mask_token_id)[1]
            if len(mask_pos) < length:
                continue
            for candidate, idx in candidates:
                curr_score = 0.0
                for i in range(length):
                    pos = mask_pos[i]
                    token_id_pos = idx[i]
                    curr_score += log_probs[0, pos, token_id_pos]

                score = curr_score / length
                if score > highest_score:
                    highest_score = score
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

    def _filter_translatable_candidates(self, candidate_pool: list) -> list:
        """Filters candidate words to those with known English entries in lexicon_map or lem_map."""
        translatable = []
        translator = str.maketrans("", "", string.punctuation)
        for word in candidate_pool:
            clean_w = str(word).strip().lower()
            stripped_w = clean_w.translate(translator)
            if (
                clean_w in self.lexicon_map
                or stripped_w in self.lexicon_map
                or (hasattr(self, 'lem_map') and clean_w in self.lem_map)
            ):
                translatable.append(word)
        return translatable if translatable else candidate_pool

    def resolve_slot_translation(self, chosen_token: str, slot_tag: str = "") -> str:
        translator = str.maketrans("", "", string.punctuation)
        clean_token = chosen_token.lower().translate(translator)
        clean_tag = re.sub(r"^_{2,4}", "", slot_tag).upper()

        if clean_token in self.lem_map:
            res = self.lem_map[clean_token]
            return res

        target_col_lem = next(
            (col for col in self.lem_seg.columns if 'english' in col.lower() or 'translation' in col.lower()),
            self.lem_seg.columns[-1]
        )
        glossing_col = next((col for col in self.lem_seg.columns if 'gloss' in col.lower() or 'prediction' in col.lower()), None)

        if glossing_col and glossing_col in self.lem_seg.columns:
            target_gloss = f"['{clean_token} {clean_tag}']"
            matched_rows = self.lem_seg[self.lem_seg[glossing_col] == target_gloss]
            if not matched_rows.empty:
                return str(matched_rows[target_col_lem].iloc[0])

        raw_analysis = self.predict_morphology(clean_token, batch_size=64)
        analysis_str = raw_analysis if isinstance(raw_analysis, str) else ""
        root_match = re.search(r"\-\s*([\w]+)$", analysis_str)
        root = root_match.group(1).lower() if root_match else ""

        if root and root in self.lem_map:
            return self.lem_map[root]

        normalized_tag = re.sub(r"^N(\d+)$", r"BANTU\1", clean_tag)
        if normalized_tag in self.grammar_map:
            return self.grammar_map[normalized_tag]
        if clean_tag in self.grammar_map:
            return self.grammar_map[clean_tag]

        return f"['{clean_tag} {clean_tag}']"

    def ranked_translation(self, fig_or_lit: str, translation_keyword: str = "translation", lang: str | None = None):
        if lang is None:
            raise ValueError("Provide a language")

        self.lang = lang
        self.nllb_tag = {
            "ganda": "lug_Latn",
            "gikuyu": "kik_Latn",
            "tshiluba": "lua_Latn",
            "chiga": "cgg_Latn",
            "tooro": "ttj_Latn",
            "runyoro": "nyo_Latn",
            "kamba": "kam_Latn"
        }
        nllb_lang = self.nllb_tag[self.lang.lower()]
        self.translation_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name, src_lang=nllb_lang)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name).to(self.device)

        self.translation_keyword = translation_keyword

        self.df_translation = pd.read_csv(
            f"{parent_path}/data/{self.lang.lower()}/{fig_or_lit}_{self.lang.lower()}_random_15 - {fig_or_lit}_{self.lang.lower()}_random_15.csv"
        )
        self.df_translation = self.df_translation[self.df_translation["who"].notna()]
        self.lexicon = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower()}_translated.csv")
        self.lem_seg = pd.read_csv(f"{parent_path}/data/{self.lang.lower()}/{self.lang.lower().capitalize()}_model_lem_seg.csv", sep='\t')

        self.lang_data = pd.DataFrame(self.data[self.data["language"] == self.lang])
        self.grammar_lookup = pd.DataFrame(self.grammar_data[self.grammar_data["language"] == self.lang])

        self.lexicon_map = self._normalize_lexicon()

        word_col_lem = 'Surface Word' if 'Surface Word' in self.lem_seg.columns else ('word' if 'word' in self.lem_seg.columns else self.lem_seg.columns[0])
        target_col_lem = 'Surface Word'
        self.lem_map = dict(zip(self.lem_seg[word_col_lem].astype(str).str.lower(), self.lexicon[target_col_lem].astype(str)))

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
            if isinstance(t, str) and re.search(r"_{2,4}(?:\.[\w\d]+)*", t):
                prov = getattr(r, "african_proverb")
                if isinstance(prov, str):
                    prov = prov.translate(translator)
                reference_pool.append({"template": t, "embedding": self._get_embedding(prov)})

        ranked_indices = []
        for row in self.df_translation.itertuples():
            translation = getattr(row, self.translation_keyword)
            current_prov = getattr(row, "african_proverb")

            raw_prov = current_prov if (current_prov is not None and not pd.isna(current_prov)) else ""
            clean_prov = str(raw_prov).translate(translator).strip()

            working_sentence = ""
            is_borrowed = False

            has_valid_translation = isinstance(translation, str) and not pd.isna(translation) and bool(translation.strip())
            has_slots_in_translation = has_valid_translation and bool(re.search(r"_{2,4}(?:\.[\w\d]+)*", translation))

            if has_valid_translation and not has_slots_in_translation:
                working_sentence = translation

            elif clean_prov and reference_pool:
                current_emb = self._get_embedding(clean_prov)
                best_template = None
                best_sim = 0.70  # Cosine similarity threshold to avoid irrelevant matches

                for ref in reference_pool:
                    similarity = torch.mm(current_emb, ref["embedding"].T).item()
                    if similarity > best_sim:
                        best_sim = similarity
                        best_template = ref["template"]
                if best_template:
                    working_sentence = best_template
                    is_borrowed = True
                elif has_valid_translation:
                    working_sentence = translation
                else:
                    working_sentence = clean_prov
            elif has_valid_translation:
                working_sentence = translation
            else:
                working_sentence = clean_prov

            slots = sorted(self.extract_slots(working_sentence), key=len, reverse=True)

            if is_borrowed:
                print(f"\n[BantuBERTa Retrieval] Borrowed template for: {current_prov}")
                print(f"Borrowed Frame: {working_sentence}")

            gloss_col = next((col for col in self.lexicon.columns if 'gloss' in col.lower()), 'Glossing')
            word_col = 'Surface Word' if 'Surface Word' in self.lexicon.columns else ('word' if 'word' in self.lexicon.columns else self.lexicon.columns[0])

            for slot_tag in slots:
                if slot_tag not in working_sentence:
                    continue

                clean_tag = re.sub(r"^_{2,4}", "", slot_tag)
                tag_components = [
                    re.sub(r"^N(\d+)$", r"BANTU\1", c.upper())
                    for c in clean_tag.split('.') if c
                ]
                candidate_pool: list = []
                if not tag_components:
                    candidate_pool = [w for w in self.lexicon[word_col].dropna().unique().tolist() if str(w).strip()]
                else:
                    mask = pd.Series(True, index=self.lexicon.index)
                    for comp in tag_components:
                        if gloss_col in self.lexicon.columns:
                            mask &= self.lexicon[gloss_col].astype(str).str.contains(comp, case=False, na=False, regex=False)
                    candidate_pool = [w for w in self.lexicon[mask][word_col].dropna().unique().tolist() if str(w).strip()]

                if not candidate_pool:
                    candidate_pool = [w for w in self.lexicon[word_col].dropna().unique().tolist() if str(w).strip()]

                candidate_pool = self._filter_translatable_candidates(candidate_pool)

                chosen_token = self._find_best_candidate(working_sentence, slot_tag, candidate_pool)

                if chosen_token:
                    english_val = self.resolve_slot_translation(chosen_token, slot_tag=slot_tag)
                    working_sentence = working_sentence.replace(slot_tag, str(english_val), 1)

            residual_slots = sorted(self.extract_slots(working_sentence), key=len, reverse=True)
            if residual_slots:
                global_fallback_pool = [w for w in self.lexicon[word_col].dropna().unique().tolist() if str(w).strip()]
                global_fallback_pool = self._filter_translatable_candidates(global_fallback_pool)

                for residual_tag in residual_slots:
                    if residual_tag not in working_sentence:
                        continue
                    fallback_token = self._find_best_candidate(working_sentence, residual_tag, global_fallback_pool)
                    if fallback_token:
                        english_val = self.resolve_slot_translation(fallback_token, residual_tag)
                        print(f"English val: {english_val}")
                        working_sentence = working_sentence.replace(residual_tag, str(english_val), 1)
                    else:
                        clean_descriptor = re.sub(r"^_{2,4}", "", residual_tag)
                        working_sentence = working_sentence.replace(residual_tag, f"[{clean_descriptor}]", 1)

            ranked_indices.append(working_sentence)

        updated_ranked_indices = []
        for sentence in ranked_indices:
            word_list = sentence.split(" ")
            for pos, word in enumerate(word_list):
                if word not in words.words():
                    replace_inout = self.translation_tokenizer(word, return_tensors="pt").to(self.device)
                    tgt_token_id = self.translation_tokenizer.convert_tokens_to_ids("eng_Latn")
                    with torch.no_grad():
                        output = self.translation_model.generate(
                            **replace_inout,
                            forced_bos_token_id=tgt_token_id,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True
                        )
                    replace_word = self.translation_tokenizer.decode(output[0], skip_special_tokens=True)
                    word_list[pos] = replace_word
            updated_ranked_indices.append(" ".join(word_list))

        return updated_ranked_indices
