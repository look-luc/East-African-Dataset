from pathlib import Path

from src.main import main

script_parent = Path(__file__).resolve().parent

if __name__ == "__main__":
    tasks = ["count_morphemes","model_segment" , "morpheme_translate"]
    languages = ["ganda", "gikuyu", "tshiluba", "chiga", "tooro", "runyoro", "kamba"]
    for task,lang in zip(tasks,languages):
        print(f"Doing {task}...")
        main(script_parent, task, lang)
    print("Ran")
