import datasets
import pandas as pd

DATASET_PATH = (
    "/home/magedsaeed/Downloads/ANERcorp-CamelLabSplits/ANERcorp-CamelLabSplits"
)
TEST_SPLIT_PATH = f"{DATASET_PATH}/ANERCorp_CamelLab_test.txt"
TRAIN_SPLIT_PATH = f"{DATASET_PATH}/ANERCorp_CamelLab_train.txt"


def prepare_split_as_data_frame(split_file):
    samples_tokens = []
    samples_tags = []
    with open(split_file, "r") as f:
        sample_tokens = []
        sample_tags = []
        for line in f:
            line = line.strip()
            if not line:
                samples_tokens.append(sample_tokens)
                samples_tags.append(sample_tags)
                sample_tokens = []
                sample_tags = []
                continue
            line_tokens = line.split()
            if len(line_tokens) > 2:
                raise ValueError(f"line {line} has more than two tokens")
            sample_tokens.append(line_tokens[0])
            sample_tags.append(line_tokens[1])
    unique_tags = set(tag for sample_tags in samples_tags for tag in sample_tags)
    tags_classes = {tag: idx for idx, tag in enumerate(unique_tags)}
    samples_tags_labels = [
        [tags_classes[tag] for tag in sample_tags] for sample_tags in samples_tags
    ]
    # sanity checks
    assert (
        len(samples_tokens) == len(samples_tags_labels) == len(samples_tags)
    ), f"different lengths, samples_tokens length: {len(samples_tokens)}, samples_tags length: {len(samples_tags)}, samples_tags_labels length: {len(samples_tags_labels)}"
    for sample_tokens, sample_tags, sample_tags_labels in zip(
        samples_tokens,
        samples_tags,
        samples_tags_labels,
    ):
        assert (
            len(sample_tokens) == len(sample_tags) == len(sample_tags_labels)
        ), f"different lengths, sample_tokens length: {len(sample_tokens)}, sample_tags length: {len(sample_tags)}, sample_tags_labels length: {len(sample_tags_labels)}"
    split_data_frame = pd.DataFrame(
        {
            "tokens": samples_tokens,
            "tags": samples_tags,
            "tags_labels": samples_tags_labels,
        }
    )
    return split_data_frame


train_data_frame = prepare_split_as_data_frame(split_file=TRAIN_SPLIT_PATH)
test_data_frame = prepare_split_as_data_frame(split_file=TEST_SPLIT_PATH)


dataset_dict = datasets.DatasetDict(
    dict(
        train=datasets.Dataset.from_pandas(train_data_frame),
        test=datasets.Dataset.from_pandas(test_data_frame),
    )
)

dataset_dict.push_to_hub(
    private=True,
    repo_id="ANERcorp",
)  # double check to make the dataset private while pushing to hub. This is not to violate the dataset providers rights
