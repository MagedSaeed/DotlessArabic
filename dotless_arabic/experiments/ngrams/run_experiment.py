import sys
import click
import tkseem as tk
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.experiments.ngrams import constants
from dotless_arabic.experiments.ngrams.src.training_pipeline import training_pipeline
from dotless_arabic.processing import process, undot


@click.command()
@click.option(
    "--tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(
        [
            "CharacterTokenizer",
            "DisjointLetterTokenizer",
            "FarasaMorphologicalTokenizer",
            "WordTokenizer",
        ]
    ),
)
@click.option(
    "--dataset",
    help="dataset name to train. Note that results files will be saved in '{current_dir_of_this_file}'/{dataset}_dataset/",
    required=True,
    type=click.Choice(type=click.Choice(list(constants.COLLECT_DATASET.keys()))),
)
def run(dataset, tokenizer_class):

    dataset_name = dataset + "_dataset"

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/results/{dataset_name}"

    # create results dir if not exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_class = getattr(tk, tokenizer_class)

    dotted_results_file_path = (
        f"{results_dir}/results_dotted_tokenizer_{tokenizer_class.__name__}.txt"
    )

    dataset = constants.COLLECT_DATASET[dataset](results_file=dotted_results_file_path)

    # delete the current logging file, if exists

    Path(dotted_results_file_path).unlink(missing_ok=True)

    log_content(
        content=f"""
            Dotted Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
            """,
        results_file=dotted_results_file_path,
    )

    dataset = list(
        map(
            process,
            tqdm(dataset),
        ),
    )

    log_content(
        content=f"""
        Some of the Dataset Samples before training:
        {constants.NEW_LINE.join(dataset[:5])}
        """,
        results_file=dotted_results_file_path,
    )

    ################################################
    ###### Dotted Dataset Training #################
    ################################################

    dataset_id = f"dotted-{dataset_name}".upper()

    training_pipeline(
        dataset=dataset,
        dataset_name=dataset_id.lower(),
        tokenizer_class=tokenizer_class,
        results_file=dotted_results_file_path,
    )

    log_content(
        content=f"""
        Dotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=dotted_results_file_path,
    )

    ################################################
    ###### Undotted Dataset Training ###############
    ################################################

    undotted_results_file_path = (
        f"{results_dir}/results_undotted_tokenizer_{tokenizer_class.__name__}.txt"
    )

    # delete the current logging file, if exists

    Path(undotted_results_file_path).unlink(missing_ok=True)

    log_content(
        content=f"""
        Undotted Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
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

    log_content(
        content=f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(undotted_dataset[:5])}
        """,
        results_file=undotted_results_file_path,
    )

    training_pipeline(
        dataset=undotted_dataset,
        dataset_name=dataset_id.lower(),
        tokenizer_class=tokenizer_class,
        results_file=undotted_results_file_path,
    )

    log_content(
        content=f"""
        Undotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=undotted_results_file_path,
    )


if __name__ == "__main__":
    run()
