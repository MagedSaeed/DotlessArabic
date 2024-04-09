import pandas as pd
import datasets


def prepare_columns(example):
    example["ar"] = example["translation"]["ar"]
    example["en"] = example["translation"]["en"]
    return example


dataset = datasets.load_dataset("iwslt2017", "iwslt2017-ar-en")

train_dataset = (
    dataset["train"]
    .filter(
        lambda example: len(example["translation"]["ar"])
        and len(example["translation"]["en"])
    )
    .map(
        prepare_columns,
        remove_columns=["translation"],
    )
    # .select(range(1_000))
)
val_dataset = (
    dataset["validation"]
    .filter(
        lambda example: len(example["translation"]["ar"])
        and len(example["translation"]["en"])
    )
    .map(
        prepare_columns,
        remove_columns=["translation"],
    )
    # .select(range(10))
)
test_dataset = (
    dataset["test"]
    .filter(
        lambda example: len(example["translation"]["ar"])
        and len(example["translation"]["en"])
    )
    .map(
        prepare_columns,
        remove_columns=["translation"],
    )
    # .select(range(1000))
    .select(
        range(len(dataset["test"]) - 1205, len(dataset["test"]))
    )  # select only 2015 test set, refer to the dataset card: (https://huggingface.co/datasets/iwslt2017/blob/c18a4f81a47ae6fa079fe9d32db288ddde38451d/iwslt2017.py#L151) and this research (https://www.researchgate.net/profile/Aliakbar-Abdurahimov/publication/368275060_NLP_project_final_report_1/links/63dea67364fc8606381a53ba/NLP-project-final-report-1.pdf)
    # ).select(range(100))
)


def collect_parallel_train_dataset_for_translation():
    return train_dataset


def collect_parallel_val_dataset_for_translation():
    return val_dataset


def collect_parallel_test_dataset_for_translation():
    return test_dataset


def collect_train_dataset_for_dots_retreival():
    return train_dataset["ar"]


def collect_val_dataset_for_dots_retrieval():
    return val_dataset["ar"]


def collect_test_dataset_for_dots_retrieval():
    return test_dataset["ar"]
