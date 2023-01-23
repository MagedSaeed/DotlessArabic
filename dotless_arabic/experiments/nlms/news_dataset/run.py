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
    process,
    undot,
)

################################################
############# Dataset Preparation ##############
################################################


current_dir = Path(__file__).resolve().parent

dotted_results_file_path = f"{current_dir}/results_dotted.txt"

# delete the current logging file, if exists

Path(dotted_results_file_path).unlink(missing_ok=True)

log_to_file(
    text=f"""
    Dotted Training Started
    """,
    results_file=dotted_results_file_path,
)


hf_news_dataset = datasets.concatenate_datasets(
    [
        datasets.load_dataset("arabic_billion_words", "Alittihad", split="train"),
    ]
)


dataset = hf_news_dataset["text"]


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

dataset_name = "news_dataset"


dataset = list(
    map(
        process,
        tqdm(dataset),
    ),
)


log_to_file(
    text=f"""
    Some of the Dataset Samples after processing:
    {constants.NEW_LINE.join(dataset[:5])}
    """,
    results_file=dotted_results_file_path,
)

################################################
###### Dotted Dataset Training #################
################################################

dataset_id = f"dotted-{dataset_name}".upper()

training_pipeline(
    dataset=dataset,
    dataset_name=dataset_id.lower(),
    dataset_id=dataset_id,
    is_dotted=True,
    results_file=dotted_results_file_path,
)

log_to_file(
    text=f"""
    Dotted Training Finished
    """,
    results_file=dotted_results_file_path,
)

################################################
###### Undotted Dataset Training ###############
################################################

undotted_results_file_path = f"{current_dir}/results_undotted.txt"

# delete the current logging file, if exists

Path(undotted_results_file_path).unlink(missing_ok=True)


log_to_file(
    text=f"""
    Undotted Training Started
    """,
    results_file=undotted_results_file_path,
)

dataset_id = f"undotted-{dataset_name}".upper()

undotted_dataset = list(
    map(
        undot,
        dataset,
    )
)


log_to_file(
    text=f"""
    Some of the Dataset Samples after undotting:
    {constants.NEW_LINE.join(undotted_dataset[:5])}
    """,
    results_file=undotted_results_file_path,
)


training_pipeline(
    dataset=undotted_dataset,
    dataset_name=dataset_id.lower(),
    dataset_id=dataset_id,
    is_dotted=False,
    results_file=undotted_results_file_path,
)

log_to_file(
    text=f"""
    Undotted Training Finished
    """,
    results_file=undotted_results_file_path,
)
