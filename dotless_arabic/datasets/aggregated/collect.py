import sys

if "." not in sys.path:
    sys.path.append(".")

from dotless_arabic.utils import log_content
from dotless_arabic.experiments.nlms.src import constants

from dotless_arabic.datasets.news.collect import (
    collect_dataset as collect_news_dataset,
)
from dotless_arabic.datasets.poems.collect import (
    collect_dataset as collect_poems_dataset,
)
from dotless_arabic.datasets.quran.collect import (
    collect_dataset as collect_quran_dataset,
)
from dotless_arabic.datasets.sanadset_hadeeth.collect import (
    collect_dataset as collect_hadeeth_dataset,
)
from dotless_arabic.datasets.wikipedia.collect import (
    collect_dataset as collect_wikipedia_dataset,
)
from dotless_arabic.processing import dataset_dot_transform, dataset_newline_transform


def collect_dataset(results_file=None):

    dataset = (
        collect_quran_dataset()
        + collect_hadeeth_dataset()
        + collect_poems_dataset()
        + collect_news_dataset()
        + collect_wikipedia_dataset()
    )

    if results_file is not None:
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

    # just to double check

    dataset = dataset_dot_transform(dataset_newline_transform(dataset))

    if results_file is not None:
        log_content(
            content=f"""
            Number of Samples after transformations:
            {len(dataset):,}
            """,
            results_file=results_file,
        )

    return dataset
