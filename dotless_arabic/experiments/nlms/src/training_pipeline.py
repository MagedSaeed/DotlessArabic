from tqdm.auto import tqdm
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.utilities.model_summary import ModelSummary

from dotless_arabic.experiments.nlms.src import constants
from dotless_arabic.experiments.nlms.src.callbacks import LossMetricsCallback
from dotless_arabic.experiments.nlms.src.models import LitNeuralLanguageModel
from dotless_arabic.experiments.nlms.src.settings import configure_environment
from dotless_arabic.experiments.nlms.src.utils import (
    calculate_perplexity,
    generate_text,
    get_sequence_length,
    get_best_checkpoint,
    get_dataloader,
    get_tokenizer,
    get_vocab_size,
    log_to_file,
    train_lm,
    get_oovs_rate,
)


def training_pipeline(
    dataset,
    is_dotted,
    dataset_id,
    batch_size,
    gpu_devices,
    cpu_devices,
    dataset_name,
    results_file,
    vocab_coverage,
    tokenizer_class,
    sequence_length=None,
    print_to_console=True,
):
    configure_environment()
    train_dataset, test_dataset = train_test_split(
        dataset,
        shuffle=True,
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_SEED,
    )

    train_dataset, val_dataset = train_test_split(
        train_dataset,
        shuffle=True,
        test_size=constants.VAL_SIZE,
        random_state=constants.RANDOM_SEED,
    )

    log_to_file(
        text=f"""
        Train Samples: {len(train_dataset):,}
        Val Samples: {len(val_dataset):,}
        Test Samples: {len(test_dataset):,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    log_to_file(
        text=f"""
        Calculating vocab size:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    vocab_size, all_vocab = get_vocab_size(
        dataset=train_dataset,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
    )
    log_to_file(
        text=f"""
        Considered Vocab: {vocab_size:,}
        All Vocab: {all_vocab:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    tokenizer = get_tokenizer(
        vocab_size=vocab_size,
        train_dataset=train_dataset,
        tokenizer_class=tokenizer_class,
    )
    log_to_file(
        text=f"""
        Tokenizer Vocab Size (to make sure): {tokenizer.vocab_size:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    log_to_file(
        text=f"""
        Calculating Sequence Length:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    if sequence_length is None:
        sequence_length = get_sequence_length(
            dataset=list(
                map(
                    tokenizer.split_text,
                    tqdm(train_dataset),
                )
            )
        )
    log_to_file(
        text=f"""
        Sequence Length: {sequence_length:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    log_to_file(
        text=f"""
        Building DataLoaders
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    train_dataloader = get_dataloader(
        shuffle=True,
        tokenizer=tokenizer,
        dataset=train_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        drop_last=constants.DEFAULT_BATCH_SIZE < len(val_dataset),
    )
    test_dataloader = get_dataloader(
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    log_to_file(
        text=f"""
        Train DataLoader: {len(train_dataloader):,}
        Val DataLoader: {len(val_dataloader):,}
        Test DataLoader: {len(test_dataloader):,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    timer_callback = Timer()

    loss_metrics_callback = LossMetricsCallback()
    lm_model = LitNeuralLanguageModel(vocab_size=tokenizer.vocab_size)

    log_to_file(
        text=f"""
        {ModelSummary(lm_model)}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    wandb_logger = WandbLogger(
        project=f"NLMs",
        id=dataset_id,
        group=dataset_name,
        job_type="dotted" if is_dotted else "undotted",
        name=dataset_name + f"_{tokenizer_class.__name__}",
    )
    trainer = train_lm(
        # one_run=True,
        lm_model=lm_model,
        dataset_id=dataset_id,
        cpu_devices=cpu_devices,
        gpu_devices=gpu_devices,
        wandb_logger=wandb_logger,
        val_dataloader=val_dataloader,
        vocab_coverage=vocab_coverage,
        max_epochs=constants.MAX_EPOCHS,
        tokenizer_class=tokenizer_class,
        train_dataloader=train_dataloader,
        callbacks=[loss_metrics_callback, timer_callback],
    )
    lm_model = LitNeuralLanguageModel.load_from_checkpoint(
        get_best_checkpoint(dataset_id=dataset_id)
    )
    training_perplexity = calculate_perplexity(
        lm_model=lm_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    perplexity_with_oovs = calculate_perplexity(
        lm_model=lm_model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        sequence_length=sequence_length,
    )

    perplexity_without_oovs = calculate_perplexity(
        lm_model=lm_model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        sequence_length=sequence_length,
        ignore_oovs=True,
    )

    log_to_file(
        text=f"""
        Training Perplexity: {training_perplexity}
        Perplexity with OOVs: {perplexity_with_oovs}
        Perplexity without OOVs: {perplexity_without_oovs:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    training_oov_rate = get_oovs_rate(dataloader=train_dataloader)
    val_oov_rate = get_oovs_rate(dataloader=train_dataloader)
    test_oov_rate = get_oovs_rate(dataloader=train_dataloader)

    log_to_file(
        text=f"""
        Training OOVs rate: {training_oov_rate}
        Validation OOVs rate: {val_oov_rate}
        Test OOVs rate: {test_oov_rate}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    f'{timer_callback.time_elapsed("train"):.2f} seconds'

    log_to_file(
        text=f"""
        Training Time: {f'{timer_callback.time_elapsed("train"):.2f} seconds'}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    prompt = "<bos> "

    log_to_file(
        text=generate_text(
            prompt=prompt,
            lm_model=lm_model,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
        ),
        results_file=results_file,
        print_to_console=print_to_console,
    )
