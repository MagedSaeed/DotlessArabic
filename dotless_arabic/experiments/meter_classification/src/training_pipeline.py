import json
import torch
import torchmetrics
from tqdm.auto import tqdm
from farasa.segmenter import FarasaSegmenter
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb
from dotless_arabic.experiments.meter_classification.src.models import (
    LitMeterClassificationModel,
)

from dotless_arabic.utils import log_content

from dotless_arabic.experiments.meter_classification.src.datasets import get_dataloader


from dotless_arabic.tokenizers import get_tokenizer
from dotless_arabic.experiments.meter_classification.src import constants
from dotless_arabic.experiments.meter_classification.src.utils import (
    # get_best_checkpoint,
    train_meter_classifier,
)
from dotless_arabic.experiments.meter_classification.src.models import (
    LitMeterClassificationModel,
)
from dotless_arabic.experiments.meter_classification.src.settings import (
    configure_environment,
)

from dotless_arabic.experiments.nlms.src.utils import (
    get_vocab_size,
    get_oovs_rate,
)


def training_pipeline(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    is_dotted,
    batch_size,
    gpu_devices,
    cpu_devices,
    dataset_name,
    results_file,
    dropout_prob,
    cap_threshold,
    vocab_coverage,
    tokenizer_class,
    sequence_length=None,
    print_to_console=True,
    dataloader_workers=constants.CPU_COUNT,
):
    configure_environment()

    log_content(
        content=f"""
        Calculating vocab size using WordTokenizer:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    vocab_size, all_vocab = get_vocab_size(
        dataset=x_train,
        undot_text=not is_dotted,
        vocab_coverage=vocab_coverage,
    )
    # if tokenizer_class != WordTokenizer:
    # add 4 to account for other special chars such as unk and pad.
    # This is severe for char tokenizer but can be okay for others.
    vocab_size += 4
    all_vocab += 4

    log_content(
        content=f"""
        Considered Vocab (from WordTokenizer): {vocab_size:,}
        All Vocab (WordTokenizer): {all_vocab:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    tokenizer = get_tokenizer(
        train_dataset=x_train,
        vocab_size=vocab_size,
        undot_text=not is_dotted,
        tokenizer_class=tokenizer_class,
    )
    log_content(
        content=f"""
        Tokenizer Vocab Size: {tokenizer.vocab_size:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    log_content(
        content=f"""
        Calculating Sequence Length:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    sequence_length = len(max(x_train, key=len))
    # sequence_length = 80

    log_content(
        content=f"""
        Sequence Length: {sequence_length:,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    log_content(
        content=f"""
        Building DataLoaders
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )
    train_dataloader = get_dataloader(
        shuffle=True,
        tokenizer=tokenizer,
        baits=x_train,
        meters=y_train,
        batch_size=batch_size,
        undot_text=not is_dotted,
        workers=dataloader_workers,
        sequence_length=sequence_length,
    )
    val_dataloader = get_dataloader(
        baits=x_val,
        meters=y_val,
        tokenizer=tokenizer,
        batch_size=batch_size,
        undot_text=not is_dotted,
        workers=dataloader_workers,
        sequence_length=sequence_length,
        drop_last=constants.DEFAULT_BATCH_SIZE < len(x_val),
    )
    test_dataloader = get_dataloader(
        baits=x_test,
        meters=y_test,
        tokenizer=tokenizer,
        batch_size=batch_size,
        undot_text=not is_dotted,
        workers=dataloader_workers,
        sequence_length=sequence_length,
    )
    # print(test_dataloader.dataset[:5])
    log_content(
        content=f"""
        Train DataLoader: {len(train_dataloader):,}
        Val DataLoader: {len(val_dataloader):,}
        Test DataLoader: {len(test_dataloader):,}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    training_oov_rate = get_oovs_rate(dataloader=train_dataloader)
    val_oov_rate = get_oovs_rate(dataloader=val_dataloader)
    test_oov_rate = get_oovs_rate(dataloader=test_dataloader)

    log_content(
        content=f"""
        Training OOVs rate: {training_oov_rate}
        Validation OOVs rate: {val_oov_rate}
        Test OOVs rate: {test_oov_rate}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    timer_callback = Timer()

    meter_classifier = LitMeterClassificationModel(
        dropout_prob=dropout_prob,
        vocab_size=tokenizer.vocab_size,
    )

    log_content(
        content=f"""
        {ModelSummary(meter_classifier)}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    text_type = "dotted" if is_dotted else "undotted"

    wandb_logger = WandbLogger(
        project=f"MC",
        name=f"{text_type}_{dataset_name}_{tokenizer_class.__name__}_{cap_threshold}",
    )
    wandb_logger.watch(meter_classifier, log="all")
    trainer = train_meter_classifier(
        text_type=text_type,
        cpu_devices=cpu_devices,
        gpu_devices=gpu_devices,
        dataset_name=dataset_name,
        wandb_logger=wandb_logger,
        cap_threshold=cap_threshold,
        val_dataloader=val_dataloader,
        train_dataloader=train_dataloader,
        max_epochs=constants.MAX_EPOCHS,
        tokenizer_class=tokenizer_class,
        meter_classifier=meter_classifier,
        callbacks=[timer_callback],
    )
    # meter_classifier = LitMeterClassificationModel.load_from_checkpoint(
    #     get_best_checkpoint(
    #         text_type=text_type,
    #         dataset_name=dataset_name,
    #         cap_threshold=cap_threshold,
    #         tokenizer_class=tokenizer_class,
    #     )
    # )

    f'{timer_callback.time_elapsed("train"):.2f} seconds'

    log_content(
        content=f"""
        Training Time: {f'{timer_callback.time_elapsed("train"):.2f} seconds'}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    results = trainer.test(
        ckpt_path="best",
        dataloaders=(
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ),
    )

    # test_accuracy = results["test_acc"]
    # test_loss = results["test_loss"]

    log_content(
        content=f"""
        Test Results: {json.dumps(results,ensure_ascii=False,indent=4,)}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    # calculate the confusion matrix
    predictions, labels = list(), list()
    for batch_predictions, batch_labels in trainer.predict(
        ckpt_path="best",
        dataloaders=test_dataloader,
    ):
        predictions.extend(batch_predictions.tolist())
        labels.extend(batch_labels.tolist())
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    test_confusion_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass",
        num_classes=len(set(labels.tolist())),
    )(predictions, labels)

    log_content(
        content=f"""
        Test Confusion Matrix:
        {test_confusion_matrix}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    wandb.finish()
