import json
import random
import nltk
from sklearn.model_selection import train_test_split
import torch
import torchmetrics
from tqdm.auto import tqdm
from farasa.stemmer import FarasaStemmer
from farasa.segmenter import FarasaSegmenter
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb
from dotless_arabic.callbacks import EpochTimerCallback
from dotless_arabic.experiments.topic_modeling.src.tuners import (
    tune_topic_modeling_model,
)


from dotless_arabic.utils import log_content
from dotless_arabic.processing import process

from dotless_arabic.experiments.topic_modeling.src.datasets import get_dataloader
from dotless_arabic.experiments.topic_modeling.src.settings import (
    configure_environment,
)
from dotless_arabic.experiments.topic_modeling.src import constants
from dotless_arabic.experiments.topic_modeling.src.utils import (
    get_balanced_sub_dataset,
    get_oovs_rate,
    train_topic_modeler,
    get_sequence_length,
    filter_empty_items,
)
from dotless_arabic.experiments.topic_modeling.src.models import (
    LitTopicModelingModel,
)

from dotless_arabic.datasets.sanad.collect import (
    collect_train_dataset_for_topic_modeling,
    collect_test_dataset_for_topic_modeling,
)

from dotless_arabic.tokenizers import get_tokenizer


from dotless_arabic.experiments.topic_modeling.src.utils import (
    get_vocab_size,
    get_oovs_rate,
)


def training_pipeline(
    is_dotted,
    batch_size,
    gpu_devices,
    results_file,
    tokenizer_class,
    best_params=None,
    print_to_console=True,
    dataloader_workers=constants.CPU_COUNT,
):
    configure_environment()

    if best_params is None:
        best_params = {}

    train_dataset = collect_train_dataset_for_topic_modeling()
    test_dataset = collect_test_dataset_for_topic_modeling()

    def shuffle_dict_items(items):
        random.shuffle(list(items))
        return items

    train_dataset = dict(shuffle_dict_items(train_dataset.items()))

    x_train, x_val, y_train, y_val = train_test_split(
        list(train_dataset.keys()),
        list(train_dataset.values()),
        test_size=0.1,
    )

    num_classes = len(set(y_train))

    log_content(
        content=f"""
        Number of classes is: {num_classes}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    x_test, y_test = list(test_dataset.keys()), list(test_dataset.values())

    log_content(
        content=f"""
        Processing texts:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    x_train = list(map(process, tqdm(x_train)))
    x_val = list(map(process, tqdm(x_val)))
    x_test = list(map(process, tqdm(x_test)))

    # log_content(
    #     content=f"""
    #     Removing Stopwords:
    #     """,
    #     results_file=results_file,
    #     print_to_console=print_to_console,
    # )

    # stopwords = list(map(process, nltk.corpus.stopwords.words("arabic")))

    # x_train = [
    #     " ".join(token for token in document.split() if token not in stopwords)
    #     for document in tqdm(x_train)
    # ]
    # x_val = [
    #     " ".join(token for token in document.split() if token not in stopwords)
    #     for document in tqdm(x_val)
    # ]
    # x_test = [
    #     " ".join(token for token in document.split() if token not in stopwords)
    #     for document in tqdm(x_test)
    # ]

    # log_content(
    #     content=f"""
    #     Stemming:
    #     """,
    #     results_file=results_file,
    #     print_to_console=print_to_console,
    # )

    # stemmer = FarasaStemmer(interactive=True)

    # x_train = [stemmer.stem(document) for document in tqdm(x_train)]
    # x_val = [stemmer.stem(document) for document in tqdm(x_val)]
    # x_test = [stemmer.stem(document) for document in tqdm(x_test)]

    log_content(
        content=f"""
        Filtering empty documents:
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    x_train, y_train = filter_empty_items(xs=x_train, ys=y_train)
    x_val, y_val = filter_empty_items(xs=x_val, ys=y_val)
    x_test, y_test = filter_empty_items(xs=x_test, ys=y_test)

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
        vocab_coverage=constants.DEFAULT_VOCAB_COVERAGE,
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
        vocab_size=vocab_size,
        train_dataset=x_train,
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

    sequence_length = get_sequence_length(
        dataset=[tokenizer.tokenize_from_splits(document) for document in x_train]
    )

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
        docs=x_train,
        labels=y_train,
        tokenizer=tokenizer,
        batch_size=batch_size,
        undot_text=not is_dotted,
        workers=dataloader_workers,
        sequence_length=sequence_length,
    )
    val_dataloader = get_dataloader(
        docs=x_val,
        labels=y_val,
        tokenizer=tokenizer,
        batch_size=batch_size,
        undot_text=not is_dotted,
        workers=dataloader_workers,
        sequence_length=sequence_length,
        drop_last=constants.DEFAULT_BATCH_SIZE < len(x_val),
    )
    test_dataloader = get_dataloader(
        docs=x_test,
        labels=y_test,
        tokenizer=tokenizer,
        batch_size=batch_size,
        undot_text=not is_dotted,
        workers=dataloader_workers,
        sequence_length=sequence_length,
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

    training_oov_rate = get_oovs_rate(
        unk_token_id=tokenizer.token_to_id(tokenizer.unk_token),
        dataloader=train_dataloader,
    )
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

    per_epoch_timer = EpochTimerCallback()

    if not best_params:
        # tune the model
        sub_x_train, sub_y_train = get_balanced_sub_dataset(
            xs=x_train,
            ys=y_train,
            classes_threshold=1000,
        )
        sub_x_val, sub_y_val = get_balanced_sub_dataset(
            xs=x_val,
            ys=y_val,
            classes_threshold=100,
        )

        sub_tokenizer = get_tokenizer(
            undot_text=False,
            vocab_size=vocab_size,
            train_dataset=sub_x_train,
            tokenizer_class=tokenizer_class,
        )

        sub_train_dataloader = get_dataloader(
            workers=1,
            shuffle=True,
            docs=sub_x_train,
            labels=sub_y_train,
            use_tqdm=False,
            undot_text=False,
            batch_size=batch_size,
            tokenizer=sub_tokenizer,
            sequence_length=get_sequence_length(dataset=sub_x_train),
        )
        sub_val_dataloader = get_dataloader(
            workers=1,
            docs=sub_x_val,
            labels=sub_y_val,
            use_tqdm=False,
            undot_text=False,
            tokenizer=sub_tokenizer,
            sequence_length=get_sequence_length(dataset=sub_x_train),
        )

        best_params = tune_topic_modeling_model(
            vocab_size=tokenizer.vocab_size,
            train_dataloader=sub_train_dataloader,
            val_dataloader=sub_val_dataloader,
        )

    topics_modeler = LitTopicModelingModel(
        num_classes=num_classes,
        vocab_size=tokenizer.vocab_size,
        **best_params,
    )

    log_content(
        content=f"""
        {ModelSummary(topics_modeler)}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    text_type = "dotted" if is_dotted else "undotted"

    wandb_logger = WandbLogger(
        project=f"TM",
        name=f"{text_type}_{tokenizer_class.__name__}",
    )
    wandb_logger.watch(topics_modeler, log="all")
    trainer = train_topic_modeler(
        text_type=text_type,
        gpu_devices=gpu_devices,
        wandb_logger=wandb_logger,
        topic_modeler=topics_modeler,
        val_dataloader=val_dataloader,
        train_dataloader=train_dataloader,
        max_epochs=constants.MAX_EPOCHS,
        tokenizer_class=tokenizer_class,
        callbacks=[timer_callback, per_epoch_timer],
    )

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
    wandb_logger.experiment.log({"epoch-avg-time": per_epoch_timer.average_epochs_time})

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
        Results: {json.dumps(results,ensure_ascii=False,indent=4)}
        """,
        results_file=results_file,
        print_to_console=print_to_console,
    )

    wandb.finish()

    return best_params

    # calculate the confusion matrix
    # predictions, labels = list(), list()
    # for batch_predictions, batch_labels in trainer.predict(
    #     ckpt_path="best",
    #     dataloaders=test_dataloader,
    # ):
    #     predictions.extend(batch_predictions.tolist())
    #     labels.extend(batch_labels.tolist())
    # predictions = torch.tensor(predictions)
    # labels = torch.tensor(labels)
    # test_confusion_matrix = torchmetrics.ConfusionMatrix(
    #     task="multiclass",
    #     num_classes=len(set(labels.tolist())),
    # )(predictions, labels)

    # log_content(
    #     content=f"""
    #     Test Confusion Matrix:
    #     {test_confusion_matrix}
    #     """,
    #     results_file=results_file,
    #     print_to_console=print_to_console,
    # )
