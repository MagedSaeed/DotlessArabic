import datasets
from dotless_arabic.utils import log_content


def collect_train_dataset_for_sentiment_analysis(results_file=None):
    dataset = datasets.load_dataset("labr")
    train_dataset = dataset["train"]

    log_content(
        content=f"""
        Number of samples in train split: {len(train_dataset)}
        """,
        results_file=results_file,
    )

    log_content(
        content=f"""
        Dropping out the neutral class (label=2)
        """,
        results_file=results_file,
    )

    train_dataset = train_dataset.filter(lambda example: example["label"] != 2)

    log_content(
        content=f"""
        Number of train samples after dropping the neutral samples: {len(train_dataset)}
        """,
        results_file=results_file,
    )

    def classes_map(example):
        if example["label"] in [0, 1]:
            example["label"] = 0
        elif example["label"] in [3, 4]:
            example["label"] = 1
        else:
            raise ValueError(
                f"{example['label']} could not be identified as positive/negative sentiment"
            )
        return example

    train_dataset = train_dataset.map(classes_map)

    log_content(
        content=f"""
        Converting dataset to a dictionary of text:label
        """,
        results_file=results_file,
    )

    return {
        text: label
        for text, label in zip(train_dataset["text"], train_dataset["label"])
    }


def collect_test_dataset_for_sentiment_analysis(results_file=None):
    dataset = datasets.load_dataset("labr")
    test_dataset = dataset["test"]

    log_content(
        content=f"""
        Number of samples in test split: {len(test_dataset)}
        """,
        results_file=results_file,
    )

    log_content(
        content=f"""
        Dropping out the neutral class (label=2)
        """,
        results_file=results_file,
    )

    test_dataset = test_dataset.filter(lambda example: example["label"] != 2)

    log_content(
        content=f"""
        Number of test samples after dropping the neutral samples: {len(test_dataset)}
        """,
        results_file=results_file,
    )

    def classes_map(example):
        if example["label"] in [0, 1]:
            example["label"] = 0
        elif example["label"] in [3, 4]:
            example["label"] = 1
        else:
            raise ValueError(
                f"{example['label']} could not be identified as positive/negative sentiment"
            )
        return example

    test_dataset = test_dataset.map(classes_map)

    log_content(
        content=f"""
        Converting dataset to a dictionary of text:label
        """,
        results_file=results_file,
    )

    return {
        text: label for text, label in zip(test_dataset["text"], test_dataset["label"])
    }

    return test_dataset
