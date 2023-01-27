import sys
import tkseem as tk
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.training_pipeline import training_pipeline
from dotless_arabic.experiments.nlms.src.utils import log_to_file
from dotless_arabic.processing import process, undot


def run(
    dataset,
    dataset_name,
    results_dir,
    vocab_coverage,
    tokenizer_class,
):
    tokenizer_class = getattr(tk, tokenizer_class)

    dotted_results_file_path = f"{results_dir}/results_dotted_tokenizer_{tokenizer_class.__name__}_vocab_coverage_{vocab_coverage}.txt"

    # delete the current logging file, if exists

    Path(dotted_results_file_path).unlink(missing_ok=True)

    log_to_file(
        text=f"""
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

    log_to_file(
        text=f"""
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
        is_dotted=True,
        dataset=dataset,
        dataset_id=dataset_id,
        vocab_coverage=vocab_coverage,
        dataset_name=dataset_id.lower(),
        tokenizer_class=tokenizer_class,
        results_file=dotted_results_file_path,
    )

    log_to_file(
        text=f"""
        Dotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=dotted_results_file_path,
    )

    ################################################
    ###### Undotted Dataset Training ###############
    ################################################

    undotted_results_file_path = f"{results_dir}/results_undotted_tokenizer_{tokenizer_class.__name__}_vocab_coverage_{vocab_coverage}.txt"

    # delete the current logging file, if exists

    Path(undotted_results_file_path).unlink(missing_ok=True)

    log_to_file(
        text=f"""
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

    log_to_file(
        text=f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(undotted_dataset[:5])}
        """,
        results_file=undotted_results_file_path,
    )

    training_pipeline(
        is_dotted=False,
        dataset_id=dataset_id,
        dataset=undotted_dataset,
        vocab_coverage=vocab_coverage,
        dataset_name=dataset_id.lower(),
        tokenizer_class=tokenizer_class,
        results_file=undotted_results_file_path,
    )

    log_to_file(
        text=f"""
        Undotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
        """,
        results_file=undotted_results_file_path,
    )
