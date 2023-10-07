import torch

import os

from dotless_arabic.experiments.constants import *

from dotless_arabic.tokenizers import WordTokenizer


MAX_EPOCHS = 250
DEFAULT_BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIGHTING_ACCELERATOR = "auto"  # used by pytorch_lightning

GPU_DEVICES = "0"  # consider only one CPU core if there is no GPU

CPU_DEVICES = 1  # I tried with higher values but this did not work

CPU_COUNT = os.cpu_count()

SEQUENCE_LENGTH_PERCENTILE = 0.85  # to neglect outliers documents in terms of length

DEFAULT_TOKENIZER_CLASS = WordTokenizer
