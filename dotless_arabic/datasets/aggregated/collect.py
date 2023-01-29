import sys
from pathlib import Path

if "." not in sys.path:
    sys.path.append(".")

from pathlib import Path

from dotless_arabic.datasets.news_dataset.collect import (
    collect_dataset as collect_news_dataset,
)
from dotless_arabic.datasets.poems_dataset.collect import (
    collect_dataset as collect_poems_dataset,
)
from dotless_arabic.datasets.quran_dataset.collect import (
    collect_dataset as collect_quran_dataset,
)
from dotless_arabic.datasets.sanadset_hadeeth_dataset.collect import (
    collect_dataset as collect_hadeeth_dataset,
)
from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.datasets.wikipedia_dataset.collect import (
    collect_dataset as collect_wikipedia_dataset,
)
from dotless_arabic.processing import dataset_dot_transform, dataset_newline_transform


def collect_dataset(log_steps=True):

    if log_steps:
        current_dir = Path(__file__).resolve().parent

        dotted_results_file_path = f"{current_dir}/results_dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

    dataset = (
        collect_quran_dataset(log_steps=False)
        + collect_hadeeth_dataset(log_steps=False)
        + collect_poems_dataset(log_steps=False)
        + collect_news_dataset(log_steps=False)
        + collect_wikipedia_dataset(log_steps=False)
    )

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

    # just to double check

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
