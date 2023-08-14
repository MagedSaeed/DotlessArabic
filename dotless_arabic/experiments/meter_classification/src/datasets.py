from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from dotless_arabic.processing import undot

from dotless_arabic.experiments.meter_classification.src import constants


class BaitsDataset(Dataset):
    def __init__(
        self,
        baits,
        meters,
        tokenizer,
        max_bait_length,
        use_tqdm=True,
        undot_text=False,
    ):
        super().__init__()
        max_length = max_bait_length
        baits = tqdm(baits) if use_tqdm else baits
        self.encoded_dataset = []
        self.X = baits
        self.y = meters
        for bait in baits:
            if not bait:
                raise
            if undot_text:
                bait = undot(bait)
            tokenized_bait = tokenizer.tokenize_from_splits(bait)
            encoded_bait = []
            for token in tokenized_bait:
                encoded_bait.append(tokenizer.token_to_id(token))
            encoded_bait = tokenizer.pad(encoded_bait, length=max_length)
            encoded_bait = encoded_bait[:max_length]
            self.encoded_dataset.append(encoded_bait)

    def __getitem__(self, index):
        inputs = torch.LongTensor(self.encoded_dataset[index])
        outputs = torch.LongTensor([self.y[index]])
        return inputs, outputs

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.encoded_dataset)


def get_dataloader(
    baits,
    meters,
    tokenizer,
    sequence_length,
    shuffle=False,
    drop_last=True,
    undot_text=False,
    workers=constants.CPU_COUNT,
    batch_size=constants.DEFAULT_BATCH_SIZE,
):
    baits_dataset = BaitsDataset(
        baits=baits,
        meters=meters,
        tokenizer=tokenizer,
        undot_text=undot_text,
        max_bait_length=sequence_length,
    )
    dataloader = DataLoader(
        shuffle=shuffle,
        dataset=baits_dataset,
        num_workers=workers,
        drop_last=drop_last,
        batch_size=batch_size,
    )
    return dataloader
