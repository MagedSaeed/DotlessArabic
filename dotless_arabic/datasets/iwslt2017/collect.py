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
    # .select(range(1000))
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
    # .select(range(10))
)


def collect_parallel_train_dataset_for_translation():
    return train_dataset


def collect_parallel_val_dataset_for_translation():
    return val_dataset


def collect_parallel_test_dataset_for_translation():
    return test_dataset
