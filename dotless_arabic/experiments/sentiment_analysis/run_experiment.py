import sys
import click
from pathlib import Path
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.tokenizers import TOKENIZERS_MAP
from dotless_arabic.experiments.sentiment_analysis.src import constants
from dotless_arabic.experiments.sentiment_analysis.src.best_hparams import best_hparams
from dotless_arabic.experiments.sentiment_analysis.src.training_pipeline import (
    training_pipeline,
)


@click.command()
@click.option(
    "--tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the dataset",
    type=click.Choice(list(TOKENIZERS_MAP.keys())),
)
@click.option(
    "--gpu_devices",
    help="GPU devices indexes to consider if the machine has GPUs. Expected input to be index,index...",
    type=str,
    default=str(constants.GPU_DEVICES),
)
@click.option(
    "--batch_size",
    help="Batch size to consider in various data setups",
    type=int,
    default=constants.DEFAULT_BATCH_SIZE,
)
@click.option(
    "--run_dotted",
    help="Run DOTTED experiment. This is True by default",
    type=bool,
    default=True,
)
@click.option(
    "--run_undotted",
    help="Run UNDOTTED experiment. This is True by default",
    type=bool,
    default=True,
)
@click.option(
    "--model_type",
    default="RNN",
    help="Model architecture",
    type=click.Choice(
        [
            "RNN",
            "Transformer",
        ],
        case_sensitive=False,
    ),
)
def run(
    tokenizer_class,
    gpu_devices,
    batch_size,
    run_dotted,
    run_undotted,
    model_type,
):
    assert (
        run_dotted or run_undotted
    ), "run_dotted and run_undotted should not be both False"

    # gpu_devices = list(map(int, gpu_devices.split(",")))

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/results/{tokenizer_class}"

    # create results dir if not exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    tokenizer_class = TOKENIZERS_MAP[tokenizer_class]

    experiment_best_hparams = best_hparams["labr"].get(
        tokenizer_class.__name__,
        {},
    )

    if run_dotted:
        dotted_results_file_path = f"{results_dir}/dotted.txt"

        # delete the current logging file, if exists

        Path(dotted_results_file_path).unlink(missing_ok=True)

    if run_undotted:
        undotted_results_file_path = f"{results_dir}/undotted.txt"

        # delete the current logging file, if exists

        Path(undotted_results_file_path).unlink(missing_ok=True)

    ################################################
    ###### Dotted Dataset Training #################
    ################################################

    if run_dotted:
        log_content(
            content=f"""
                Dotted Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
                """,
            results_file=dotted_results_file_path,
        )

        experiment_best_hparams = training_pipeline(
            is_dotted=True,
            batch_size=batch_size,
            gpu_devices=gpu_devices,
            tokenizer_class=tokenizer_class,
            best_hparams=experiment_best_hparams,
            results_file=dotted_results_file_path,
        )

        log_content(
            content=f"""
            Dotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
            """,
            results_file=dotted_results_file_path,
        )
    else:
        print(
            "#" * 100,
            f"""
            Dotted Experiment is DISABLED for tokenizer {tokenizer_class.__name__}
            """,
            "#" * 100,
        )

    ################################################
    ###### Undotted Dataset Training ###############
    ################################################

    if run_undotted:
        log_content(
            content=f"""
            Undotted Training Started at {datetime.now()} for tokenizer: {tokenizer_class.__name__}
            """,
            results_file=undotted_results_file_path,
        )

        training_pipeline(
            is_dotted=False,
            batch_size=batch_size,
            gpu_devices=gpu_devices,
            tokenizer_class=tokenizer_class,
            best_hparams=experiment_best_hparams,
            results_file=undotted_results_file_path,
        )

        log_content(
            content=f"""
            Undotted Training Finished for tokenizer {tokenizer_class.__name__} at {datetime.now()}
            """,
            results_file=undotted_results_file_path,
        )
    else:
        print(
            "#" * 100,
            f"""
            UNDotted Experiment is DISABLED for tokenizer {tokenizer_class.__name__}
            """,
            "#" * 100,
        )


if __name__ == "__main__":
    run()
