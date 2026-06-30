from pathlib import Path

from src.main import main

script_parent = Path(__file__).resolve().parent

if __name__ == "__main__":
    main(script_parent, "count_morphemes")
    print("Ran")
