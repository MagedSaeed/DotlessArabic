import sys
import click
from pathlib import Path
from datetime import datetime


if "." not in sys.path:
    sys.path.append(".")


from dotless_arabic.utils import log_content
from dotless_arabic.tokenizers import TOKENIZERS_MAP
from dotless_arabic.experiments.translation.src import constants
from dotless_arabic.experiments.translation.src.training_pipeline import (
    training_pipeline,
)


@click.command()
@click.option(
    "--source_tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the source dataset",
    type=click.Choice(["WordTokenizer", "SentencePieceTokenizer"]),
)
@click.option(
    "--target_tokenizer_class",
    default=constants.DEFAULT_TOKENIZER_CLASS,
    help="Tokenizer class to tokenize the target dataset",
    type=click.Choice(["WordTokenizer", "SentencePieceTokenizer"]),
)
@click.option(
    "--source_lang",
    help="Source Language",
    type=click.Choice(["en", "ar"]),
)
@click.option(
    "--target_lang",
    help="Target Language",
    type=click.Choice(["en", "ar"]),
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
def run(
    source_tokenizer_class,
    target_tokenizer_class,
    source_lang,
    target_lang,
    gpu_devices,
    batch_size,
    run_dotted,
    run_undotted,
):
    assert (
        run_dotted or run_undotted
    ), "run_dotted and run_undotted should not be both False"

    # gpu_devices = list(map(int, gpu_devices.split(",")))

    current_dir = Path(__file__).resolve().parent

    results_dir = f"{current_dir}/results/{source_lang}_to_{target_lang}/{source_tokenizer_class}_to_{target_tokenizer_class}"

    # create results dir if not exists
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    source_tokenizer_class = TOKENIZERS_MAP[source_tokenizer_class]
    target_tokenizer_class = TOKENIZERS_MAP[target_tokenizer_class]

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
    best_params = {}

    if run_dotted:
        log_content(
            content=f"""
                Dotted Training Started at {datetime.now()} for tokenizer: {source_tokenizer_class.__name__}
                """,
            results_file=dotted_results_file_path,
        )

        best_params = training_pipeline(
            is_dotted=True,
            batch_size=batch_size,
            gpu_devices=gpu_devices,
            source_language_code=source_lang,
            target_language_code=target_lang,
            results_file=dotted_results_file_path,
            source_tokenizer_class=source_tokenizer_class,
            target_tokenizer_class=target_tokenizer_class,
        )

        log_content(
            content=f"""
            Dotted Training Finished for tokenizer {source_tokenizer_class.__name__} at {datetime.now()}
            """,
            results_file=dotted_results_file_path,
        )
    else:
        print(
            "#" * 100,
            f"""
            Dotted Experiment is DISABLED for tokenizer {source_tokenizer_class.__name__}
            """,
            "#" * 100,
        )

    ################################################
    ###### Undotted Dataset Training ###############
    ################################################

    if run_undotted:
        log_content(
            content=f"""
            Undotted Training Started at {datetime.now()} for tokenizer: {source_tokenizer_class.__name__}
            """,
            results_file=undotted_results_file_path,
        )

        training_pipeline(
            is_dotted=False,
            batch_size=batch_size,
            gpu_devices=gpu_devices,
            best_params=best_params,
            source_language_code=source_lang,
            target_language_code=target_lang,
            results_file=undotted_results_file_path,
            source_tokenizer_class=source_tokenizer_class,
            target_tokenizer_class=target_tokenizer_class,
        )

        log_content(
            content=f"""
            Undotted Training Finished for tokenizer {source_tokenizer_class.__name__} at {datetime.now()}
            """,
            results_file=undotted_results_file_path,
        )
    else:
        print(
            "#" * 100,
            f"""
            UNDotted Experiment is DISABLED for tokenizer {source_tokenizer_class.__name__}
            """,
            "#" * 100,
        )


if __name__ == "__main__":
    run()
