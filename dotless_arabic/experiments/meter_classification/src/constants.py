import torch

from dotless_arabic.experiments.constants import *

from dotless_arabic.tokenizers import CharacterTokenizer


GRU_LAYERS = 5  # for GRU
GRU_DROPOUT = 0.25

MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 512
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 256
LEARNING_RATE = 0.001
DEFAULT_VOCAB_COVERAGE = (
    1  # for tokenizer to consider only tokens that cover this value of the running text
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIGHTING_ACCELERATOR = "auto"  # used by pytorch_lightning

TEST_SIZE = 0.1
VAL_SIZE = 0.05

GPU_DEVICES = "0"  # consider only one CPU core if there is no GPU

CPU_DEVICES = 1  # I tried with higher values but this did not work

DEFAULT_TOKENIZER_CLASS = CharacterTokenizer
