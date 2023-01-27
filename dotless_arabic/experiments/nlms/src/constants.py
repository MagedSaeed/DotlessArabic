import tkseem as tk
import torch

from dotless_arabic.constants import *

DEFAULT_TOKENIZER_CLASS = "WordTokenizer"

NUM_LAYERS = 4  # for GRU


MAX_EPOCHS = 100
BATCH_SIZE = 256
HIDDEN_SIZE = 400
DROPOUT_PROB = 0.333
EMBEDDING_SIZE = 400
LEARNING_RATE = 0.001
DEFAULT_VOCAB_COVERAGE = 0.95  # for tokenizer to consider only tokens that cover this value of the running text
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_SIZE = 0.1
VAL_SIZE = 0.05

GPU_DEVICES = [0]
