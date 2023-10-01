import math
import re

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from dotless_arabic.experiments.translation.src import constants


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 5000,
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_token_id=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token_id)
        # self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=constants.DEVICE)) == 1).transpose(
        0, 1
    )
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=constants.DEVICE)) == 1).transpose(
        0, 1
    )
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=constants.DEVICE).type(
        torch.bool
    )
    src_padding_mask = (src == pad_idx).clone().detach()
    tgt_padding_mask = (tgt == pad_idx).clone().detach()
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class TranslationTransformer(LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        nhead=8,
        emb_size=256,
        pad_token_id=1,
        num_decoder_layers=2,
        num_encoder_layers=2,
        dim_feedforward=2048,
        dropout: float = 0.1,
        learning_rate=0.0001,
        # src_vocab_size=source_tokenizer.vocab_size,
        # tgt_vocab_size=target_tokenizer.vocab_size,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_token_id = pad_token_id
        self.learning_rate = learning_rate
        self.source_vocab_size = src_vocab_size
        self.target_vocab_size = tgt_vocab_size

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout=dropout,
        )
        self.dense = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, trg):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src,
            trg,
            pad_idx=self.pad_token_id,
        )
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.dense(outs)

    def step(self, batch):
        inputs, targets = batch
        outputs = self(inputs, targets[:, :-1].contiguous())
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.step(batch)
        loss = F.cross_entropy(
            outputs.view(-1, self.target_vocab_size),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
        )
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.step(batch)
        loss = F.cross_entropy(
            outputs.view(-1, self.target_vocab_size),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        outputs = self.step(batch)
        loss = F.cross_entropy(
            outputs.view(-1, self.target_vocab_size),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
        )
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.25,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def translate(
        self,
        input_sentence,
        source_tokenizer,
        target_tokenizer,
        max_sequence_length,
    ):
        was_training = self.training is True
        self.eval()
        encoded_input_sentence = (
            torch.tensor(source_tokenizer.encode(input_sentence))
            .view(1, -1)
            .to(constants.DEVICE)
        )
        target = "<bos> "
        for i in range(
            max_sequence_length,
        ):
            encoded_target = (
                torch.tensor(target_tokenizer.encode(target))
                .view(1, -1)
                .to(constants.DEVICE)
            )
            outputs = self(src=encoded_input_sentence, trg=encoded_target)
            next_word_id = torch.argmax(outputs[0, i, :])
            next_word = target_tokenizer.id_to_token(next_word_id)
            target += f"{next_word.strip()} "
            if next_word == "<eos>":
                break
        if was_training:
            self.train()
        target = re.sub("\s+", " ", target).strip()
        return target