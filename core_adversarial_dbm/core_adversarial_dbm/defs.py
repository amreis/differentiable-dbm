import pathlib

import torch as T

ROOT_PATH = pathlib.Path(__file__).parent.parent.parent
DEVICE = "cuda" if T.cuda.is_available() else "cpu"
