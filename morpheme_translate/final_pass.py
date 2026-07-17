from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import model_info
from transformers import AutoModel, AutoTokenizer

parent_path = script_dir = Path(__file__).resolve().parent.parent

class translation_final_pass:
    def __init__(
        self,
        model:str="dsfsi/BantuBERTa",
        data_file:str=f"{parent_path}/data/data.csv",
        grammar_file:str=f"{parent_path}/data/bantu_grammar_lookup.csv",
        lang:str|None=None,
    ) -> None:
        self.data = pd.read_csv(data_file, sep='\t')
        self.grammar_data = pd.read_csv(grammar_file)

        self.model_name = model
