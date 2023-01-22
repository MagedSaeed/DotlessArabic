import os
import sys
from pathlib import Path

import datasets
import pandas as pd
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.datasets import dataset_dot_transform
from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.processing import process, undot
from dotless_arabic.utils import download_file

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

ashaar = datasets.load_dataset("arbml/ashaar", split="train")

baits = list()

for poem in ashaar["poem verses"]:
    baits.extend(poem)

baits = list(set(baits))


log_to_file(
    text=f"""
    Sample of datasets samples:
    {constants.NEW_LINE.join(baits[:5])}
    """,
    results_file=dotted_results_file_path,
)

log_to_file(
    text=f"""
    Number of Baits:
    {len(baits):,}
    """,
    results_file=dotted_results_file_path,
)

baits = list(
    filter(
        lambda bait: 25 > len(bait.replace(" ", "")) > 5,
        baits,
    )
)

log_to_file(
    text=f"""
    Number of baits after deleting 25>bait>5 chars:
    {len(baits):,}
    """,
    results_file=dotted_results_file_path,
)

dataset_name = "poems_dataset"
dataset_id = f"dotted-{dataset_name}".upper()

dataset = baits

dataset = list(
    map(
        process,
        tqdm(dataset),
    ),
)


log_to_file(
    text=f"""
    Some of the Dataset Samples before training:
    {constants.NEW_LINE.join(dataset[:5])}
    """,
    results_file=dotted_results_file_path,
)

################################################
###### Dotted Dataset Training #################
################################################

training_pipeline(
    dataset=dataset,
    dataset_name="dotted-quran_dataset",
    dataset_id="dotted-qurand_dataset".upper(),
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
    Some of the Dataset Samples before training:
    {constants.NEW_LINE.join(undotted_dataset[:5])}
    """,
    results_file=undotted_results_file_path,
)


training_pipeline(
    dataset=undotted_dataset,
    dataset_name=dataset_name,
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
