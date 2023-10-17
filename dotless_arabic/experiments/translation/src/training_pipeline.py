import json
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb
from tqdm.auto import tqdm

import mosestokenizer
from farasa.segmenter import FarasaSegmenter

from dotless_arabic.callbacks import EpochTimerCallback
from dotless_arabic.experiments.translation.src.tuners import (
    tune_translation_model,
)
from dotless_arabic.processing.processing import undot

from dotless_arabic.utils import log_content

from dotless_arabic.experiments.translation.src.processing import (
    process_en,
    process_ar,
)

from dotless_arabic.experiments.translation.src.datasets import get_dataloader
from dotless_arabic.experiments.translation.src.settings import (
    configure_environment,
)
from dotless_arabic.experiments.translation.src import constants
from dotless_arabic.experiments.translation.src.utils import (
    get_oovs_rate,
    train_translator,
    get_source_tokenizer,
    get_target_tokenizer,
    get_blue_score,
    get_sequence_length,
)

from dotless_arabic.experiments.translation.src.models import (
    TranslationTransformer,
)

# from dotless_arabic.datasets.open_subtitles_arbml_subset.collect import (
#     collect_parallel_train_dataset_for_translation,
#     collect_parallel_test_dataset_for_translation,
# )

# from dotless_arabic.datasets.ted_multi.collect import (
#     collect_parallel_train_dataset_for_translation,
#     collect_parallel_test_dataset_for_translation,
#     collect_parallel_val_dataset_for_translation,
# )

from dotless_arabic.datasets.iwslt2017.collect import (
    collect_parallel_train_dataset_for_translation,
    collect_parallel_val_dataset_for_translation,
    collect_parallel_test_dataset_for_translation,
)


def training_pipeline(
    is_dotted,
    batch_size,
    gpu_devices,
    results_file,
    source_tokenizer_class,
    target_tokenizer_class,
    source_language_code,
    target_language_code,
    best_params=None,
    print_to_console=True,
):
    configure_environment()

    if best_params is None:
        best_params = {}

    text_type = "dotted" if is_dotted else "undotted"

    wandb_logger = WandbLogger(
        project=f"NMT",
        name=f"{source_language_code}_to_{target_language_code}_{text_type}_{source_tokenizer_class.__name__}_to_{target_tokenizer_class.__name__}",
    )

    # source_language_code = "en"
    # target_language_code = "ar"

    log_content(
        content=f"""
        Collecting dataset splits:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    train_dataset = collect_parallel_train_dataset_for_translation().to_pandas()
    val_dataset = collect_parallel_val_dataset_for_translation().to_pandas()
    test_dataset = collect_parallel_test_dataset_for_translation().to_pandas()

    log_content(
        content=f"""
        Processing source and target sequences:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    train_dataset["en"] = list(
        map(
            process_en,
            tqdm(train_dataset["en"]),
        )
    )
    val_dataset["en"] = list(
        map(
            process_en,
            tqdm(val_dataset["en"]),
        )
    )
    test_dataset["en"] = list(
        map(
            process_en,
            tqdm(test_dataset["en"]),
        )
    )

    train_dataset["ar"] = list(
        map(
            process_ar,
            tqdm(train_dataset["ar"]),
        )
    )
    val_dataset["ar"] = list(
        map(
            process_ar,
            tqdm(val_dataset["ar"]),
        )
    )
    test_dataset["ar"] = list(
        map(
            process_ar,
            tqdm(test_dataset["ar"]),
        )
    )

    # if tokenizer_class == WordTokenizer:
    #     log_content(
    #         content=f"""
    #         Segmenting arabic with farasa:
    #         """,
    #         results_file=results_file,
    #         print_to_console=print_to_console,
    #     )

    #     segmenter = FarasaSegmenter(interactive=True)

    #     train_dataset["ar"] = train_dataset["ar"].progress_map(
    #         lambda text: segmenter.segment(text).replace("+", "+ ")
    #     )
    #     val_dataset["ar"] = val_dataset["ar"].progress_map(
    #         lambda text: segmenter.segment(text).replace("+", "+ ")
    #     )
    #     test_dataset["ar"] = test_dataset["ar"].progress_map(
    #         lambda text: segmenter.segment(text).replace("+", "+ ")
    #     )

    log_content(
        content=f"""
        Segmenting english with moses:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    with mosestokenizer.MosesTokenizer("en") as moses_tokenizer:
        train_dataset["en"] = train_dataset["en"].progress_map(
            lambda text: " ".join(moses_tokenizer(text))
        )
        val_dataset["en"] = val_dataset["en"].progress_map(
            lambda text: " ".join(moses_tokenizer(text))
        )
        test_dataset["en"] = test_dataset["en"].progress_map(
            lambda text: " ".join(moses_tokenizer(text))
        )

    if not is_dotted:
        log_content(
            content=f"""
            Undot Arabic text
            """,
            results_file=results_file,
            print_to_console=print_to_console,
        )

        train_dataset["ar"] = list(
            map(
                undot,
                tqdm(train_dataset["ar"]),
            )
        )
        val_dataset["ar"] = list(
            map(
                undot,
                tqdm(val_dataset["ar"]),
            )
        )
        test_dataset["ar"] = list(
            map(
                undot,
                tqdm(test_dataset["ar"]),
            )
        )

    log_content(
        content=f"""
        Building source and target tokenizers
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    source_tokenizer = get_source_tokenizer(
        train_dataset=train_dataset,
        tokenizer_class=source_tokenizer_class,
        source_language_code=source_language_code,
        undot_text=not is_dotted if source_language_code == "ar" else False,
    )
    target_tokenizer = get_target_tokenizer(
        train_dataset=train_dataset,
        tokenizer_class=target_tokenizer_class,
        target_language_code=target_language_code,
        undot_text=not is_dotted if target_language_code == "ar" else False,
    )

    log_content(
        content=f"""
        Source vocab size: {source_tokenizer.vocab_size}
        Target vocab size: {target_tokenizer.vocab_size}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    wandb_logger.experiment.log({"source_vocab_size": source_tokenizer.vocab_size})
    wandb_logger.experiment.log({"target_vocab_size": target_tokenizer.vocab_size})

    log_content(
        content=f"""
        Calculate sequence length:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    source_max_sequence_length = get_sequence_length(
        dataset=list(
            map(
                source_tokenizer.split_text,
                tqdm(train_dataset[source_language_code]),
            )
        )
    )
    target_max_sequence_length = get_sequence_length(
        dataset=list(
            map(
                target_tokenizer.split_text,
                tqdm(train_dataset[target_language_code]),
            )
        )
    )

    log_content(
        content=f"""
        Source Max Doc Length: {source_max_sequence_length}
        Target Max Doc Length: {target_max_sequence_length}
        """,
        # setting the sequence length to be the avg of the two
        results_file=results_file,
        print_to_console=print_to_console,
    )

    wandb_logger.experiment.log({"source-sequence-length": source_max_sequence_length})
    wandb_logger.experiment.log({"target-sequence-length": target_max_sequence_length})

    log_content(
        content=f"""
        Building DataLoaders
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    # train_dataset = train_dataset[:100]
    train_dataloader = get_dataloader(
        shuffle=True,
        batch_size=batch_size,
        dataset=train_dataset,
        undot_text=not is_dotted,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        source_sequence_length=source_max_sequence_length,
        target_sequence_length=target_max_sequence_length,
    )
    val_dataloader = get_dataloader(
        shuffle=False,
        dataset=val_dataset,
        batch_size=batch_size,
        undot_text=not is_dotted,
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        source_sequence_length=source_max_sequence_length,
        target_sequence_length=target_max_sequence_length,
    )

    test_dataloader = get_dataloader(
        shuffle=False,
        dataset=test_dataset,
        batch_size=batch_size,
        undot_text=not is_dotted,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        source_sequence_length=source_max_sequence_length,
        target_sequence_length=target_max_sequence_length,
    )

    log_content(
        content=f"""
        Train DataLoader: {len(train_dataloader):,}
        Val DataLoader: {len(val_dataloader):,}
        Test DataLoader: {len(test_dataloader):,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    # training_oov_rate = get_oovs_rate(
    #     unk_token_id=tokenizer.token_to_id(tokenizer.unk_token),
    #     dataloader=train_dataloader,
    # )
    # val_oov_rate = get_oovs_rate(dataloader=val_dataloader)
    # test_oov_rate = get_oovs_rate(dataloader=test_dataloader)

    # log_content(
    #     content=f"""
    #     Training OOVs rate: {training_oov_rate}
    #     Validation OOVs rate: {val_oov_rate}
    #     Test OOVs rate: {test_oov_rate}
    #     """,
    #     results_file=results_file,
    #     print_to_console=print_to_console,
    # )

    # wandb_logger.experiment.log({"training_oov_rate": training_oov_rate})
    # wandb_logger.experiment.log({"val_oov_rate": val_oov_rate})
    # wandb_logger.experiment.log({"test_oov_rate": test_oov_rate})

    timer_callback = Timer()

    per_epoch_timer = EpochTimerCallback()

    assert source_tokenizer.token_to_id(
        source_tokenizer.pad_token
    ) == target_tokenizer.token_to_id(target_tokenizer.pad_token)
    translator = TranslationTransformer(
        src_vocab_size=source_tokenizer.vocab_size,
        tgt_vocab_size=target_tokenizer.vocab_size,
        pad_token_id=source_tokenizer.token_to_id(source_tokenizer.pad_token),
    )

    log_content(
        content=f"""
        {ModelSummary(translator)}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    wandb_logger.watch(translator, log="all")
    validate_and_fit = False
    trainer = train_translator(
        text_type=text_type,
        translator=translator,
        gpu_devices=gpu_devices,
        wandb_logger=wandb_logger,
        val_dataloader=val_dataloader,
        max_epochs=constants.MAX_EPOCHS,
        source_lang=source_language_code,
        target_lang=target_language_code,
        train_dataloader=train_dataloader,
        validate_and_fit=validate_and_fit,
        callbacks=[timer_callback, per_epoch_timer],
        source_tokenizer_class=source_tokenizer_class,
        target_tokenizer_class=target_tokenizer_class,
    )

    if validate_and_fit:
        log_content(
            content=f"""
            Training Time: {f'{timer_callback.time_elapsed("train"):.2f} seconds'}
            """,
            results_file=results_file,
            print_to_console=print_to_console,
        )

        log_content(
            content=f"""
            Average training Time for one epoch: {f'{per_epoch_timer.average_epochs_time:.3f} seconds'}
            """,
            results_file=results_file,
            print_to_console=print_to_console,
        )
        wandb_logger.experiment.log(
            {"epoch-avg-time": per_epoch_timer.average_epochs_time}
        )

        results = trainer.test(
            ckpt_path="best",
            dataloaders=(
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ),
        )

        log_content(
            content=f"""
            Losses: {json.dumps(results,ensure_ascii=False,indent=4)}
            """,
            results_file=results_file,
            print_to_console=print_to_console,
        )

    blue_score = get_blue_score(
        model=TranslationTransformer.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
            if validate_and_fit
            else f"NMT/{text_type}/{source_tokenizer_class.__name__}/checkpoints/last.ckpt"  # take the last if validate_and_fit is False
        ).to(constants.DEVICE),
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_sequence_length=target_max_sequence_length,
        source_sentences=test_dataset[source_language_code],
        target_sentences=test_dataset[target_language_code],
    )

    log_content(
        content=f"""
        Test blue score (sacre): {blue_score}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    wandb_logger.experiment.log({"test-sacre-blue-score": blue_score})

    wandb.finish()

    return best_params
