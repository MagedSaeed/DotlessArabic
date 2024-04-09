import datasets


dataset = datasets.load_dataset("bentrevett/multi30k")

train_dataset = dataset["train"].filter(
    lambda example: len(example["de"]) and len(example["en"])
)
val_dataset = dataset["validation"].filter(
    lambda example: len(example["de"]) and len(example["en"])
)
test_dataset = dataset["test"].filter(
    lambda example: len(example["de"]) and len(example["en"])
)


def collect_parallel_train_dataset_for_translation():
    return train_dataset


def collect_parallel_val_dataset_for_translation():
    return val_dataset


def collect_parallel_test_dataset_for_translation():
    return test_dataset
