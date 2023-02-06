import sys
import click
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.processing import process, undot
from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.datasets.utils import (
    get_unique_samples_count,
    get_unique_vocab_count,
    get_all_tokens_count,
)


@click.command()
@click.option(
    "--dataset",
    help="dataset name to collect statistics for. Note that statistics files will be saved in '{current_dir_of_this_file}'/{dataset}_dataset/",
    required=True,
    type=click.Choice(list(constants.COLLECT_DATASET.keys())),
)
def run(dataset):

    dataset_name = dataset + "_dataset"

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/{dataset}"

    # create results dir if not exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    statistics_file_path = f"{results_dir}/statistics.txt"

    # delete the current logging file, if exists

    Path(statistics_file_path).unlink(missing_ok=True)

    dataset = constants.COLLECT_DATASET[dataset](results_file=statistics_file_path)

    log_content(
        content=f"""
            Dotted Statistics Analysis Started at {datetime.now()} for dataset {dataset_name}
            """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
            Processing Dataset
            """,
        results_file=statistics_file_path,
    )

    dataset = list(
        map(
            process,
            tqdm(dataset),
        ),
    )

    log_content(
        content=f"""
        Some of the Dataset Samples before collecting statistics:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
        results_file=statistics_file_path,
    )

    ################################################
    ###### Dotted Dataset statistics ###############
    ################################################

    log_content(
        content=f"""
        Dotted Statistics Analysis Finished for dataset {dataset_name} at {datetime.now()}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Samples Count: {get_unique_samples_count(dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Unique Vocabulary Count: {get_unique_vocab_count(dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        All Tokens Count: {get_all_tokens_count(dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        vocab/tokens: {get_unique_vocab_count(dataset)/get_all_tokens_count(dataset):.3f}
        """,
        results_file=statistics_file_path,
    )

    ################################################
    ###### Undotted Dataset statistics #############
    ################################################

    log_content(
        content=f"""
        Undotted Statistics Analysis Started at {datetime.now()} for dataset {dataset_name}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
            Undotting Dataset
            """,
        results_file=statistics_file_path,
    )

    undotted_dataset = list(
        map(
            undot,
            tqdm(dataset),
        )
    )

    log_content(
        content=f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(undotted_dataset[:5])}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Samples Count: {get_unique_samples_count(undotted_dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Unique Vocabulary Count: {get_unique_vocab_count(undotted_dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        All Tokens Count: {get_all_tokens_count(undotted_dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        vocab/tokens: {get_unique_vocab_count(undotted_dataset)/get_all_tokens_count(undotted_dataset):.3f}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        dotted voacb - undotted vocab: {get_unique_vocab_count(dataset)-get_unique_vocab_count(undotted_dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Undotted Statistics Analysis Finished for dataset {dataset_name} at {datetime.now()}
        """,
        results_file=statistics_file_path,
    )


if __name__ == "__main__":
    run()
