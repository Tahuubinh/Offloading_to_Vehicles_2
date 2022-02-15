import os
from pathlib import Path

LINK_PROJECT = Path(os.path.abspath(__file__))
LINK_PROJECT = LINK_PROJECT.parent.parent
#print(LINK_PROJECT)
DATA_DIR = os.path.join(LINK_PROJECT, "data")
RESULT_DIR = os.path.join(LINK_PROJECT, "result/result1/")
DATA_TASK = os.path.join(LINK_PROJECT, "data_task/200 normal task 900 - 1000")

NUM_ACTIONS = 4
class Config:
    Pr = 46
    Pr2 = 24
    Wm = 10
    n_unit_in_layer=[16, 32, 32, 8]
    length_hidden_layer = len(n_unit_in_layer)
    
