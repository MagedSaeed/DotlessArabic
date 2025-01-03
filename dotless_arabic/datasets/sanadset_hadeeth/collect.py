import os
import sys
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic import constants
from dotless_arabic.utils import log_content
from dotless_arabic.utils import download_file
from dotless_arabic.processing import dataset_dot_transform


def collect_dataset_for_analysis(results_file=None):
    current_dir = Path(__file__).resolve().parent

    if "sanadset.csv" not in os.listdir(current_dir):
        download_file(
            file_url="https://data.mendeley.com/public-files/datasets/5xth87zwb5/files/fe0857fa-73af-4873-b652-60eb7347b811/file_downloaded",
            file_name="sanadset.csv",
            output_dir=current_dir,
        )

    sanadset_hadeeth_dataset = list(
        pd.read_csv(f"{current_dir}/sanadset.csv")["Matn"].dropna()
    )
    log_content(
        content=f"""
        Original Number of Samples:
        {len(sanadset_hadeeth_dataset):,}
        """,
        results_file=results_file,
    )
    sanadset_hadeeth_dataset = sorted(list(dict.fromkeys(sanadset_hadeeth_dataset)))
    log_content(
        content=f"""
        Number of Samples after dropping duplicates:
        {len(sanadset_hadeeth_dataset):,}
        """,
        results_file=results_file,
    )
    return sanadset_hadeeth_dataset


def collect_dataset_for_language_modeling(results_file=None):

    sanadset_hadeeth_dataset = collect_dataset_for_analysis(results_file=results_file)

    log_content(
        content=f"""
        Sample of datasets documents:
        {constants.NEW_LINE.join(sanadset_hadeeth_dataset[:5])}
        """,
        results_file=results_file,
    )

    dataset = dataset_dot_transform(tqdm(sanadset_hadeeth_dataset))

    log_content(
        content=f"""
        Number of Samples after splitting on dots:
        {len(dataset):,}
        """,
        results_file=results_file,
    )

    return dataset
