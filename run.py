from pathlib import Path

from src.main import main

script_parent = Path(__file__).resolve().parent

if __name__ == "__main__":
    tasks = ["model_segment" , "morpheme_translate"]

    for task in tasks:
        main(script_parent, task)
    print("Ran")
