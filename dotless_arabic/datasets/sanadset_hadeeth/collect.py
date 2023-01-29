import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import download_file
from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.processing import dataset_dot_transform
from dotless_arabic.experiments.nlms.src.utils import log_to_file


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

    if "sanadset.csv" not in os.listdir(current_dir):
        download_file(
            file_url="https://data.mendeley.com/public-files/datasets/5xth87zwb5/files/fe0857fa-73af-4873-b652-60eb7347b811/file_downloaded",
            file_name="sanadset.csv",
            output_dir=current_dir,
        )

    sanadset_hadeeth_dataset = list(
        pd.read_csv(f"{current_dir}/sanadset.csv")["Matn"].dropna()
    )
    sanadset_hadeeth_dataset = sorted(list(set(sanadset_hadeeth_dataset)))

    if log_steps:
        log_to_file(
            text=f"""
            Sample of datasets documents:
            {constants.NEW_LINE.join(sanadset_hadeeth_dataset[:5])}
            """,
            results_file=dotted_results_file_path,
        )

        log_to_file(
            text=f"""
            Original Number of Samples:
            {len(sanadset_hadeeth_dataset):,}
            """,
            results_file=dotted_results_file_path,
        )

    dataset = list(set(sanadset_hadeeth_dataset))

    if log_steps:
        log_to_file(
            text=f"""
            Number of Samples after dropping duplicates:
            {len(dataset):,}
            """,
            results_file=dotted_results_file_path,
        )

    dataset = dataset_dot_transform(tqdm(sanadset_hadeeth_dataset))

    if log_steps:
        log_to_file(
            text=f"""
            Number of Samples after splitting on dots:
            {len(dataset):,}
            """,
            results_file=dotted_results_file_path,
        )

    return dataset
