import datasets

ted = datasets.load_dataset("ted_multi")


ted = ted.filter(
    lambda example: "ar" in example["translations"]["language"]
    and "en" in example["translations"]["language"]
)


def transform_to_en_ar(example):
    example["en"] = example["translations"]["translation"][
        example["translations"]["language"].index("en")
    ]
    example["ar"] = example["translations"]["translation"][
        example["translations"]["language"].index("ar")
    ]
    return example


ted = ted.map(transform_to_en_ar, remove_columns=["translations", "talk_name"])


def collect_parallel_train_dataset_for_translation():
    return ted["train"]


def collect_parallel_val_dataset_for_translation():
    return ted["validation"]


def collect_parallel_test_dataset_for_translation():
    return ted["test"]
