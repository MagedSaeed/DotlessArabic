import os
import random

import torch

from pytorch_lightning import seed_everything

from dotless_arabic.experiments.meter_classification.src import constants


def configure_environment():
    random.seed(constants.RANDOM_SEED)
    os.environ["WANDB_MODE"] = "disabled"  # to disable wandb logging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # to see CUDA errors
    torch.cuda.empty_cache()  # to free gpu memory
    seed_everything(42, workers=True)
    # other options: https://stackoverflow.com/questions/15197286/how-can-i-flush-gpu-memory-using-cuda-physical-reset-is-unavailable
