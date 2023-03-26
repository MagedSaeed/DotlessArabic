import sys

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic import constants
from dotless_arabic.utils import log_content
from dotless_arabic.processing import dataset_dot_transform, dataset_newline_transform


def collect_dataset_for_analysis(results_file=None):

    with open("../news_dataset.txt", "r") as news_dataset_file:
        news_dataset = news_dataset_file.read().splitlines()
    news_dataset = list(dict.fromkeys(news_dataset))
    return news_dataset


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

    return dataset
