import pandas as pd
import datasets

train_dataset = datasets.Dataset.from_pandas(
    pd.DataFrame(
        {
            "en": open("dotless_arabic/datasets/mttt/ted_train_en-ar.tok.clean.en")
            .read()
            .splitlines(),
            "ar": open("dotless_arabic/datasets/mttt/ted_train_en-ar.tok.clean.ar")
            .read()
            .splitlines(),
        }
    )
)
val_dataset = datasets.Dataset.from_pandas(
    pd.DataFrame(
        {
            "en": open("dotless_arabic/datasets/mttt/ted_dev_en-ar.tok.en")
            .read()
            .splitlines(),
            "ar": open("dotless_arabic/datasets/mttt/ted_dev_en-ar.tok.ar")
            .read()
            .splitlines(),
        }
    )
)
test_dataset = datasets.Dataset.from_pandas(
    pd.DataFrame(
        {
            "en": open("dotless_arabic/datasets/mttt/ted_test1_en-ar.tok.en")
            .read()
            .splitlines(),
            "ar": open("dotless_arabic/datasets/mttt/ted_test1_en-ar.tok.ar")
            .read()
            .splitlines(),
        }
    )
)
# ).select(range(100))


def collect_parallel_train_dataset_for_translation():
    return train_dataset


def collect_parallel_val_dataset_for_translation():
    return val_dataset


def collect_parallel_test_dataset_for_translation():
    return test_dataset
