import sys
from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic import constants
from dotless_arabic.utils import log_content
from dotless_arabic.processing import (
    dataset_dot_transform,
    dataset_newline_transform,
)


def collect_dataset_for_analysis(results_file=None):

    with open("../wikipedia_dataset.txt", "r") as news_dataset_file:
        wikipedia_dataset = news_dataset_file.read().splitlines()

    dataset = list(dict.fromkeys(wikipedia_dataset))
    return dataset


def collect_dataset_for_language_modeling(results_file=None):
    dataset = collect_dataset_for_analysis(results_file=results_file)

    log_content(
        content=f"""
        Sample of datasets samples:
        {constants.NEW_LINE.join(dataset[:2])}
        """,
        results_file=results_file,
    )

    log_content(
        content=f"""
        Number of Samples before transformations:
        {len(dataset):,}
        """,
        results_file=results_file,
    )

    dataset = dataset_dot_transform(dataset_newline_transform(dataset))

    log_content(
        content=f"""
        Number of Samples after transformations:
        {len(dataset):,}
        """,
        results_file=results_file,
    )

    # consider only those samples who have 10 tokens or more

    dataset = list(filter(lambda document: len(document.split()) >= 30, tqdm(dataset)))

    log_content(
        content=f"""
        Number of Samples when considering sample with 30 tokens or more:
        {len(dataset):,}
        """,
        results_file=results_file,
    )

    return dataset
