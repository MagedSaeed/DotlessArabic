import os
import sys
import json
import click
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.experiments import constants
from dotless_arabic.tokenizers import TOKENIZERS_MAP
from dotless_arabic.processing import process, undot
from dotless_arabic.datasets.utils import (
    tokens_frequency,
    calculate_entropy,
    tokenize_dataset_for_statistics,
)


@click.command()
@click.option(
    "--dataset",
    help="dataset name to collect statistics for. Note that statistics files will be saved in '{current_dir_of_this_file}'/{dataset}_dataset/",
    required=True,
    type=click.Choice(list(constants.COLLECT_DATASET_FOR_ANALYSIS.keys())),
)
@click.option(
    "--tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(list(TOKENIZERS_MAP.keys())),
)
@click.option(
    "--redo_counts",
    default=False,
    type=bool,
    help="Redo all tokens frequencies counts calculations from scratch i.e. do not use the cached frequencies in the json files",
)
def run(dataset, tokenizer_class, redo_counts):

    dataset_name = dataset + "_dataset"

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/{dataset}"

    tokenizer_class = TOKENIZERS_MAP[tokenizer_class]

    if redo_counts is True:
        log_content(
            content=f"""
            redo_count is True, removing previous counts files, if they exists
            """,
        )
        Path(
            f"{results_dir}/tokens_count/dotted_{tokenizer_class.__name__}.json"
        ).unlink(missing_ok=True)
        Path(
            f"{results_dir}/tokens_count/undotted_{tokenizer_class.__name__}.json"
        ).unlink(missing_ok=True)

    # create results dir if not exists
    Path(f"{results_dir}/statistics").mkdir(parents=True, exist_ok=True)

    # create tokens count dir if not exists
    Path(f"{results_dir}/tokens_count").mkdir(parents=True, exist_ok=True)

    statistics_file_path = f"{results_dir}/statistics/{tokenizer_class.__name__}.txt"

    # delete the current logging file, if exists

    Path(statistics_file_path).unlink(missing_ok=True)

    dataset = constants.COLLECT_DATASET_FOR_ANALYSIS[dataset](
        results_file=statistics_file_path
    )

    log_content(
        content=f"""
            Process the Dataset
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
            Tokenize the Dataset with {tokenizer_class.__name__}
            """,
        results_file=statistics_file_path,
    )

    dataset = tokenize_dataset_for_statistics(
        dataset=tuple(dataset),
        tokenizer_class=tokenizer_class,
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
            Dotted Statistics Analysis Started at {datetime.now()} for dataset {dataset_name} tokenized by {tokenizer_class.__name__}
            """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Samples Count: {len(dataset):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Checking if tokens count file for {tokenizer_class.__name__} exists
        """,
    )

    if os.path.isfile(
        f"{results_dir}/tokens_count/dotted_{tokenizer_class.__name__}.json"
    ):
        log_content(
            content=f"""
            Tokens count file exists, loading it..
            """,
        )
        counter = json.load(
            open(f"{results_dir}/tokens_count/dotted_{tokenizer_class.__name__}.json")
        )

    else:
        log_content(
            content=f"""
            Tokens count file does not exist, creating one and saving it..
            """,
        )
        counter = tokens_frequency(dataset=tuple(dataset))
        json.dump(
            dict(
                sorted(
                    counter.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            open(
                f"{results_dir}/tokens_count/dotted_{tokenizer_class.__name__}.json",
                "w",
            ),
            ensure_ascii=False,
            indent=4,
        )

    log_content(
        content=f"""
        Unique Vocabulary Count: {len(counter.keys()):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        All Tokens Count: {sum(counter.values()):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        vocab/tokens: {len(counter.keys())/sum(counter.values()):,.4f}
        """,
        results_file=statistics_file_path,
    )

    entropy = calculate_entropy(tokens_frequency=counter)

    log_content(
        content=f"""
        Tokens Entropy: {entropy:,.4f}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Average tokens length: {sum(map(len,counter.keys()))/len(counter.keys()):,.4f}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Dotted Statistics Analysis Finished for dataset {dataset_name} tokenized by {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=statistics_file_path,
    )

    ################################################
    ###### Undotted Dataset statistics #############
    ################################################

    log_content(
        content=f"""
        Undotted Statistics Analysis Started at {datetime.now()} for dataset {dataset_name} tokenized by {tokenizer_class.__name__}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Checking if undotted tokens count file for {tokenizer_class.__name__} exists
        """,
    )

    if os.path.isfile(
        f"{results_dir}/tokens_count/undotted_{tokenizer_class.__name__}.json"
    ):
        log_content(
            content=f"""
            Tokens count file exists, loading it..
            """,
        )
        undotted_counter = json.load(
            open(f"{results_dir}/tokens_count/undotted_{tokenizer_class.__name__}.json")
        )

    else:
        log_content(
            content=f"""
            undotted tokens count file does not exists, undotting the dataset, building one, and saving it,,
        """,
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

        undotted_counter = tokens_frequency(dataset=tuple(undotted_dataset))
        json.dump(
            dict(
                sorted(
                    undotted_counter.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            open(
                f"{results_dir}/tokens_count/undotted_{tokenizer_class.__name__}.json",
                "w",
            ),
            ensure_ascii=False,
            indent=4,
        )

    log_content(
        content=f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(map(undot,dataset[:5]))}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Unique Vocabulary Count: {len(undotted_counter.keys()):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        All Undotted Tokens Count: {sum(undotted_counter.values()):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        undotted vocab/undotted tokens: {len(undotted_counter.keys())/sum(undotted_counter.values()):,.4f}
        """,
        results_file=statistics_file_path,
    )

    undotted_entropy = calculate_entropy(tokens_frequency=undotted_counter)

    log_content(
        content=f"""
        Undotted Tokens Entropy: {undotted_entropy:.4f}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Average tokens length: {sum(map(len,undotted_counter.keys()))/len(undotted_counter.keys()):,.4f}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        dotted voacb - undotted vocab: {len(counter.keys())-len(undotted_counter.keys()):,}
        """,
        results_file=statistics_file_path,
    )

    log_content(
        content=f"""
        Undotted Statistics Analysis Finished for dataset {dataset_name} tokenized by {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=statistics_file_path,
    )


if __name__ == "__main__":
    run()
