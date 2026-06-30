import sys
from pathlib import Path

import pandas as pd

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data.get_data import Get_Data


def main():
    data = Get_Data()


    data.to_csv("data.csv", sep='\t')
    print("gathered data into data.csv")
