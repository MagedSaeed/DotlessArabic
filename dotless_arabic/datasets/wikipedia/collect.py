import os
import sys
from pathlib import Path

import datasets
import pandas as pd
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


import datasets

from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.processing import (
    dataset_dot_transform,
    dataset_newline_transform,
)


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

    with open("../wikipedia_dataset.txt", "r") as news_dataset_file:
        wikipedia_dataset = news_dataset_file.read().splitlines()

    dataset = wikipedia_dataset

    log_to_file(
        text=f"""
        Sample of datasets samples:
        {constants.NEW_LINE.join(dataset[:2])}
        """,
        results_file=dotted_results_file_path,
    )

    log_to_file(
        text=f"""
        Number of Samples before transformations:
        {len(dataset):,}
        """,
        results_file=dotted_results_file_path,
    )

    dataset = dataset_dot_transform(dataset_newline_transform(dataset))

    log_to_file(
        text=f"""
        Number of Samples after transformations:
        {len(dataset):,}
        """,
        results_file=dotted_results_file_path,
    )

    # consider only those samples who have 10 tokens or more

    dataset = list(filter(lambda document: len(document.split()) >= 30, tqdm(dataset)))

    log_to_file(
        text=f"""
        Number of Samples when considering sample with 30 tokens or more:
        {len(dataset):,}
        """,
        results_file=dotted_results_file_path,
    )

    return dataset