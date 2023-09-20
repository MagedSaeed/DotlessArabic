from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from dotless_arabic.processing import undot

from dotless_arabic.experiments.sentiment_analysis.src import constants


class NewsPapersDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        tokenizer,
        sequence_length,
        use_tqdm=True,
        undot_text=False,
    ):
        super().__init__()
        max_length = sequence_length
        X = tqdm(X) if use_tqdm else X
        self.encoded_dataset = []
        self.X = X
        self.y = y
        for review in X:
            if not review:
                raise ValueError(f"Review {review} is empty")
            if undot_text:
                review = undot(review)
            tokenized_review = tokenizer.tokenize_from_splits(review)
            encoded_review = []
            for token in tokenized_review:
                encoded_review.append(tokenizer.token_to_id(token))
            encoded_review = tokenizer.pad(encoded_review, length=max_length)
            encoded_review = encoded_review[:max_length]
            self.encoded_dataset.append(encoded_review)

    def __getitem__(self, index):
        inputs = torch.LongTensor(self.encoded_dataset[index])
        outputs = torch.LongTensor([self.y[index]], dtype=torch.float32)
        return inputs, outputs

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.encoded_dataset)


def get_dataloader(
    docs,
    labels,
    tokenizer,
    sequence_length,
    use_tqdm=True,
    shuffle=False,
    drop_last=True,
    undot_text=False,
    workers=constants.CPU_COUNT,
    batch_size=constants.DEFAULT_BATCH_SIZE,
):
    reviews_dataset = NewsPapersDataset(
        X=docs,
        y=labels,
        use_tqdm=use_tqdm,
        tokenizer=tokenizer,
        undot_text=undot_text,
        sequence_length=sequence_length,
    )
    dataloader = DataLoader(
        shuffle=shuffle,
        dataset=reviews_dataset,
        num_workers=workers,
        drop_last=drop_last,
        batch_size=batch_size,
    )
    return dataloader
