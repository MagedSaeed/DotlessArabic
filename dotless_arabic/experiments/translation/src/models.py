import heapq
import math
from typing import Any
import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import torchmetrics

from dotless_arabic.experiments.translation.src import constants
from dotless_arabic.experiments.translation.src.utils import (
    create_features_from_text_list,
)


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
#         )
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
#         """
#         """
#          we need to change the shape since we are considering batch_first=True
#          this class seems to consider batch_first=False
#         """

#         x = x.view(x.size(1), x.size(0), x.size(2))
#         x = x + self.pe[: x.size(0), :]
#         x = x.view(x.size(1), x.size(0), x.size(2))
#         return self.dropout(x)


# return x


# https://songstudio.info/tech/tech-25/
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5_000, dropout=0.1):
#         super().__init__()
#         # Make initial positional encoding matrix with 0
#         pe_matrix = torch.zeros(max_len, d_model)  # (L, d_model)
#         self.d_model = d_model

#         # Calculating position encoding values
#         for pos in range(max_len):
#             for i in range(d_model):
#                 if i % 2 == 0:
#                     pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
#                 elif i % 2 == 1:
#                     pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

#         pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
#         self.positional_encoding = pe_matrix.requires_grad_(False)

#     def forward(self, x, device):
#         x = x * math.sqrt(self.d_model)  # (B, L, d_model)
#         self.positional_encoding = self.positional_encoding.to(device)
#         x = x + self.positional_encoding[:, x.size(1), :]  # (B, L, d_model)
#         return x


# class PositionalEncoding(nn.Module):
#     def __init__(
#         self,
#         emb_size: int,
#         dropout: float,
#         max_len: int = 5000,
#     ):
#         super().__init__()
#         den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
#         pos = torch.arange(0, max_len).reshape(max_len, 1)
#         pos_embedding = torch.zeros((max_len, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)

#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer("pos_embedding", pos_embedding)

#     def forward(self, token_embedding: torch.Tensor):
#         return self.dropout(
#             token_embedding + self.pos_embedding[: token_embedding.size(0), :]
#         )


# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
        # return x


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_token_id=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_token_id)
        # self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        # return self.embedding(tokens) * math.sqrt(self.emb_size)
        # return self.embedding(tokens)


def create_masks(src, tgt, pad_idx):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
        tgt_seq_len,
        device=constants.DEVICE,
        dtype=torch.bool,
    )
    src_mask = torch.zeros(
        (src_seq_len, src_seq_len),
        device=constants.DEVICE,
        dtype=torch.bool,
    )
    src_padding_mask = (src == pad_idx).clone().detach()
    tgt_padding_mask = (tgt == pad_idx).clone().detach()
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TranslationTransformer(LightningModule):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        nhead=8,
        emb_size=512,
        pad_token_id=1,
        num_decoder_layers=6,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout: float = 0.1,
        learning_rate=0.0001,
        # tie_weights=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_token_id = pad_token_id
        self.learning_rate = learning_rate
        self.source_vocab_size = src_vocab_size
        self.target_vocab_size = tgt_vocab_size

        self.emb_size = emb_size

        self.train_ppl = torchmetrics.text.Perplexity(ignore_index=pad_token_id)
        self.val_ppl = torchmetrics.text.Perplexity(ignore_index=pad_token_id)
        self.test_ppl = torchmetrics.text.Perplexity(ignore_index=pad_token_id)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            # norm_first=True,
            batch_first=True,
        )
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.dense = nn.Linear(emb_size, tgt_vocab_size)

        # wieghts tying
        # if tie_weights:
        #     self.tgt_tok_emb.weight = self.dense.weight

    def forward(self, src, trg):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(
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
            # memory_mask=src_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            # memory_key_padding_mask=src_padding_mask,
            tgt_is_causal=True,
            # memory_is_causal=True,
        )
        return self.dense(outs)

    def step(self, batch):
        # encoder_inputs, decoder_inputs, decoder_targets = batch
        inputs, targets = batch
        # outputs = self(encoder_inputs, decoder_inputs[:, :-1])
        outputs = self(inputs, targets[:, :-1].contiguous())
        return outputs

    def training_step(self, batch, batch_idx):
        # encoder_inputs, decoder_inputs, decoder_targets = batch
        inputs, targets = batch
        outputs = self.step(batch)
        loss = F.cross_entropy(
            outputs.view(-1, self.target_vocab_size),
            # decoder_targets[:, 1:].contiguous().view(-1),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
            label_smoothing=0.1,
        )
        train_ppl = self.train_ppl(
            outputs,
            targets[:, 1:],
        )
        self.log("train_ppl", train_ppl, prog_bar=True)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # encoder_inputs, decoder_inputs, decoder_targets = batch
        inputs, targets = batch
        outputs = self.step(batch)
        loss = F.cross_entropy(
            outputs.view(-1, self.target_vocab_size),
            # decoder_targets[:, 1:].contiguous().view(-1),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
            label_smoothing=0.1,
        )
        val_ppl = self.val_ppl(
            outputs,
            targets[:, 1:],
        )
        self.log("val_ppl", val_ppl, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # encoder_inputs, decoder_inputs, decoder_targets = batch
        inputs, targets = batch
        outputs = self.step(batch)
        loss = F.cross_entropy(
            outputs.view(-1, self.target_vocab_size),
            # decoder_targets[:, 1:].contiguous().view(-1),
            targets[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
            label_smoothing=0.1,
        )
        test_ppl = self.test_ppl(
            outputs,
            targets[:, 1:],
        )
        self.log("test_ppl", test_ppl, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    # return optimizer
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer=optimizer,
    #         max_lr=self.learning_rate * 10,
    #         epochs=20,  # hypothetically
    #         steps_per_epoch=4_000,  # hypothetically
    #         pct_start=0.05,
    #         div_factor=5,  # it was 5
    #         verbose=True,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": lr_scheduler,
    #         "monitor": "val_loss",
    #     }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     lr_scheduler = CosineWarmupScheduler(
    #         optimizer=optimizer,
    #         warmup=100,
    #         max_iters=4000,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": lr_scheduler,
    #         "monitor": "val_loss",
    #     }

    # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.learning_rate,
            # eps=10**-6,
            eps=10**-9,
            betas=(0.9, 0.98),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.25,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            # "monitor": "val_loss",
            "monitor": "val_ppl",
        }

    # https://github.com/TigerJeffX/Transformers-in-pytorch/blob/main/script/run_nmt_task.py
    # def configure_optimizers(self):
    #     # optimizer = torch.optim.Adam(
    #     optimizer = torch.optim.RAdam(
    #         self.parameters(),
    #         lr=0.5,
    #         betas=(0.9, 0.98),
    #         eps=1e-9,
    #     )

    #     def _rate(step, model_size, factor, warmup):
    #         if step == 0:
    #             step = 1
    #         lr = factor * (
    #             model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    #         )
    #         # if step % 250 == 0:
    #         #     print("new lr:", lr)
    #         return lr

    #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer=optimizer,
    #         lr_lambda=lambda step: _rate(
    #             step=step,
    #             model_size=self.emb_size,
    #             factor=1.0,
    #             # warmup=400,
    #             warmup=4000,
    #         ),
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": lr_scheduler,
    #         # "monitor": "val_loss",
    #         "monitor": "val_ppl",
    #     }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         # lr=self.learning_rate,
    #         lr=1,
    #         betas=(0.9, 0.98),
    #         eps=1e-9,
    #     )

    #     def rate(step, model_size, factor, warmup):
    #         """
    #         we have to default the step to 1 for LambdaLR function
    #         to avoid zero raising to negative power.
    #         """
    #         if step == 0:
    #             step = 1
    #         lr = factor * (
    #             model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    #         )
    #         return lr

    #     scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer=optimizer,
    #         lr_lambda=lambda step: rate(
    #             step=step,
    #             model_size=self.emb_size,
    #             factor=1,
    #             warmup=4_000,
    #         ),
    #         verbose=False,
    #     )
    #     lr_scheduler_config = dict(
    #         scheduler=scheduler,
    #         interval="step",
    #         frequency=1,
    #         monitor="val_ppl",
    #     )
    #     return dict(
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler_config,
    #     )

    # return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": scheduler,
    #     # "monitor": "val_loss",
    #     "monitor": "val_ppl",
    #     "interval": "step",
    #     "frequency": 1,
    # }

    def translate(
        self,
        input_sentence,
        source_tokenizer,
        target_tokenizer,
        max_sequence_length,
        # temperature=0.1,
        temperature=0,
    ):
        self.eval()
        with torch.no_grad():
            encoded_input_sentence = (
                torch.tensor(
                    create_features_from_text_list(
                        use_tqdm=False,
                        text_list=[input_sentence],
                        tokenizer=source_tokenizer,
                        sequence_length=max_sequence_length,
                    )
                )
                .view(1, -1)
                .to(self.device)
            )
            target = "<bos> "
            encoded_target = [
                target_tokenizer.token_to_id(token) for token in target.split()
            ]
            for i in range(max_sequence_length):
                outputs = self(
                    src=encoded_input_sentence,
                    trg=torch.LongTensor(encoded_target).view(1, -1).to(self.device),
                )
                outputs = outputs[:, -1, :]
                # outputs = F.softmax(outputs, dim=-1)
                if temperature > 0:
                    outputs = torch.softmax(outputs / temperature, dim=-1)
                    next_word_id = torch.multinomial(outputs, num_samples=1).item()
                else:
                    next_word_id = torch.argmax(outputs).item()
                encoded_target.append(next_word_id)
                next_word = target_tokenizer.id_to_token(next_word_id)
                target += f"{next_word.strip()} "
                if next_word == "<eos>":
                    break
        # target = re.sub("\s+", "", target).strip()
        # target = target_tokenizer.detokenize(target)
        # target = re.sub("\s+", " ", target).strip()
        # return " ".join(target_tokenizer.decode(target_tokenizer.encode(target)))
        return (
            target_tokenizer.detokenize(target_tokenizer.decode(encoded_target))
            .replace("<bos>", "<bos> ")
            .replace("<eos>", " <eos>")
            .strip()
        )

    # using chatGPT: https://chat.openai.com/share/6f08f84c-c953-47c3-87bf-d4ed1ba6dcd6
    # some other links to consider:
    # https://stackoverflow.com/questions/64356953/batch-wise-beam-search-in-pytorch
    # https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py

    def translate_with_beam_search(
        self,
        input_sentence,
        source_tokenizer,
        target_tokenizer,
        max_sequence_length,
        beam_width=4,
        length_penalty_alpha=0.6,
        minimum_penalty_tokens_length=5,
    ):
        self.eval()
        with torch.no_grad():
            encoded_input_sentence = (
                torch.tensor(
                    create_features_from_text_list(
                        add_bos=True,
                        add_eos=True,
                        use_tqdm=False,
                        text_list=[input_sentence],
                        tokenizer=source_tokenizer,
                        sequence_length=max_sequence_length,
                    )
                )
                .view(1, -1)
                .to(self.device)
            )

            # Initialize the beam with a single empty sequence
            beams = [{"tokens": [target_tokenizer.token_to_id("<bos>")], "score": 0.0}]

            for i in range(max_sequence_length):
                next_beams = []

                for beam in beams:
                    last_token = beam["tokens"][-1]

                    if last_token == target_tokenizer.token_to_id("<eos>"):
                        # If the sequence is already completed, keep it in the beam
                        next_beams.append(beam)
                        continue

                    # Expand the current beam
                    outputs = self(
                        src=encoded_input_sentence,
                        trg=torch.LongTensor(beam["tokens"])
                        .view(1, -1)
                        .to(self.device),
                    )

                    outputs = F.softmax(outputs, dim=-1)

                    # Get the top-k token candidates
                    topk_scores, topk_indices = torch.topk(
                        outputs[:, -1, :], beam_width
                    )

                    for k in range(beam_width):
                        next_token_id = topk_indices[0, k].item()
                        next_token_score = torch.log(topk_scores[0, k]).item()

                        new_beam = {
                            "tokens": beam["tokens"] + [next_token_id],
                            "score": beam["score"] + next_token_score,
                        }

                        next_beams.append(new_beam)

                # Select the top-k beams to keep, considering beam tokens length
                next_beams = heapq.nlargest(
                    beam_width,
                    next_beams,
                    key=lambda x: x["score"]
                    / (
                        (len(x["tokens"]) + minimum_penalty_tokens_length)
                        ** length_penalty_alpha
                    ),
                )

                beams = next_beams

            # Select the best beam as the final translation
            # best_beam = max(beams, key=lambda x: x["score"])
            best_beam = max(
                beams,
                key=lambda x: x["score"]
                / (
                    (len(x["tokens"]) + minimum_penalty_tokens_length)
                    ** length_penalty_alpha
                ),
            )
            translated_tokens = best_beam["tokens"]

        # Convert token IDs to tokens and clean up the sequence
        translated_tokens = [
            target_tokenizer.id_to_token(token_id) for token_id in translated_tokens
        ]
        translated_sequence = target_tokenizer.detokenize(translated_tokens).strip()
        translated_sequence = (
            translated_sequence.replace("<bos>", "<bos> ")
            .replace("<eos>", " <eos>")
            .strip()
        )
        return translated_sequence

    # def translate_with_beam_search(
    #     self,
    #     input_sentence,
    #     source_tokenizer,
    #     target_tokenizer,
    #     max_sequence_length,
    #     beam_width=4,
    #     length_penalty_alpha=0.6,
    #     minimum_penalty_tokens_length=0,
    # ):
    #     with torch.no_grad():
    #         encoded_input_sentence = (
    #             torch.tensor(
    #                 create_features_from_text_list(
    #                     add_bos=True,
    #                     add_eos=True,
    #                     use_tqdm=False,
    #                     text_list=[input_sentence],
    #                     tokenizer=source_tokenizer,
    #                 )
    #             )
    #             .view(1, -1)
    #             .to(self.device)
    #         )
    #         beams = [([], 0)]  # (sequence, log_prob)
    #         while len(beams[0][0]) < max_sequence_length:
    #             new_beams = []
    #             for beam_seq, beam_score in beams:
    #                 target = "<bos> " + " ".join(
    #                     [target_tokenizer.id_to_token(token) for token in beam_seq]
    #                 )
    #                 encoded_target = [
    #                     target_tokenizer.token_to_id(token) for token in target.split()
    #                 ]
    #                 outputs = self(
    #                     src=encoded_input_sentence,
    #                     trg=torch.LongTensor(encoded_target)
    #                     .view(1, -1)
    #                     .to(self.device),
    #                 )
    #                 outputs = outputs[:, -1, :]
    #                 log_probs, word_ids = torch.topk(outputs, beam_width)
    #                 for i in range(beam_width):
    #                     new_seq = beam_seq + [word_ids[0][i].item()]
    #                     new_score = beam_score + log_probs[0][i].item()
    #                     new_beams.append((new_seq, new_score))
    #             beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    #             if beams[0][0][-1] == target_tokenizer.token_to_id("<eos>"):
    #                 break
    #         best_sequence = beams[0][0]
    #         return (
    #             target_tokenizer.detokenize(target_tokenizer.decode(best_sequence))
    #             .replace("<bos>", "<bos> ")
    #             .replace("<eos>", " <eos>")
    #             .strip()
    #         )
