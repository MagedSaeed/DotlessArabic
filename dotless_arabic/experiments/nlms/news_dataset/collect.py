import sys
from pathlib import Path

if "." not in sys.path:
    sys.path.append(".")

from pathlib import Path

from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.processing import dataset_dot_transform, dataset_newline_transform


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

    with open("../news_dataset.txt", "r") as news_dataset_file:
        news_dataset = news_dataset_file.read().splitlines()

    dataset = news_dataset

    if log_steps:
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

    if log_steps:
        log_to_file(
            text=f"""
            Number of Samples after transformations:
            {len(dataset):,}
            """,
            results_file=dotted_results_file_path,
        )

    return dataset
