from pathlib import Path

from src.main import main

script_parent = Path(__file__).resolve().parent

if __name__ == "__main__":
    tasks = ["get_data", "count_morphemes", "model_segment", "morpheme_translate"]
    languages = ["gikuyu", "tshiluba", "chiga", "tooro", "runyoro", "kamba"]

    # Run each language through the entire pipeline pipeline sequence
    for lang in languages:
        print(f"\n--- Processing Language: {lang.upper()} ---")
        for task in tasks:
            print(f"Executing step: {task} for {lang}...")
            main(script_parent, task, lang)

    print("Multilingual pipeline processing complete.")
