import os

from pytorch_lightning import seed_everything

import wandb
from dotless_arabic.experiments.nlms.src import constants


def configure_environment(gpu_devices="0"):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # os.environ["WANDB_MODE"] = "disabled"
    seed_everything(constants.RANDOM_SEED, workers=True)
    wandb.login()
