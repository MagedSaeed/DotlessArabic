import torch

from dotless_arabic.experiments.constants import *

from dotless_arabic.tokenizers import WordTokenizer


MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_VOCAB_COVERAGE = 0.9

SEQUENCE_LENGTH_PERCENTILE = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIGHTING_ACCELERATOR = "auto"  # used by pytorch_lightning

GPU_DEVICES = "0"  # consider only one CPU core if there is no GPU

CPU_DEVICES = 1  # I tried with higher values but this did not work

DEFAULT_TOKENIZER_CLASS = WordTokenizer
