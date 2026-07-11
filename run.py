from pathlib import Path

import torch

from src.main import main

script_parent = Path(__file__).resolve().parent

torch.set_num_threads(8)
torch.set_num_interop_threads(8)

if __name__ == "__main__":
    # tasks = ["count_morphemes", "model_segment", "morpheme_translate"]
    languages = ["ganda", "gikuyu", "tshiluba", "chiga", "tooro", "runyoro", "kamba"]
    tasks = ["model_segment","morpheme_translate"]
    # languages = ["tooro", "tshiluba"]

    # Run each language through the entire pipeline pipeline sequence
    for lang in languages:
        print(f"\n--- Processing Language: {lang.upper()} ---")
        for task in tasks:
            print(f"Executing step: {task} for {lang}...")
            main(script_parent, task, lang, "data.csv")
    # main(script_parent, "get_data", "ganda")

    print("Multilingual pipeline processing complete.")
