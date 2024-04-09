import math
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
import torchmetrics

from dotless_arabic.experiments.nlms.src import constants


class LitRNNLM(LightningModule):
    def __init__(
        self,
        vocab_size,
        pad_token_id=1,
        tie_weights=True,
        model_type=constants.RNN_TYPE,
        num_layers=constants.NUM_LAYERS,
        hidden_size=constants.HIDDEN_SIZE,
        dropout_prob=constants.DROPOUT_PROB,
        learning_rate=constants.LEARNING_RATE,
        embedding_size=constants.EMBEDDING_SIZE,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_token_id = pad_token_id

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=self.pad_token_id,
        )
        if model_type.lower() == "lstm".lower():
            self.rnn = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout_prob,
                batch_first=True,
            )
        elif model_type.lower() == "gru".lower():
            self.rnn = nn.GRU(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout_prob,
                batch_first=True,
            )
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.LeakyReLU()
        self.first_dense_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
        )
        self.second_dense_layer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
        )

        self.train_ppl = torchmetrics.text.Perplexity(ignore_index=self.pad_token_id)
        self.val_ppl = torchmetrics.text.Perplexity(ignore_index=self.pad_token_id)
        self.test_ppl = torchmetrics.text.Perplexity(ignore_index=self.pad_token_id)

        # weights tieing
        if tie_weights:
            if not self.embedding_size == self.hidden_size:
                print(
                    f"Cannot tie wight as embedding size is nto equal hidden size: {self.embedding_size}!={self.hidden_size}"
                )
            else:
                # assert (
                #     self.embedding_size == self.hidden_size
                # ), "in weights tieing, embedding size should be the same as hidden size"
                self.second_dense_layer.weight = self.embedding_layer.weight

    def forward(self, x, hiddens=None):
        outputs = self.embedding_layer(x)
        # adding dropout to embeddings
        outputs = self.dropout_layer(outputs)
        # pack sequences to remove pad tokens (effecient training)
        inputs_lengths = torch.sum(
            x != self.pad_token_id,
            axis=-1,
        ).cpu()
        packed_outputs = nn.utils.rnn.pack_padded_sequence(
            outputs,
            inputs_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hiddens = self.rnn(packed_outputs, hiddens)
        outputs, lengths = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            # sequence length is the second dim, first dim is the batch size
            total_length=x.size(1),
        )
        outputs = self.first_dense_layer(outputs)
        outputs = self.dropout_layer(outputs)
        outputs = self.relu(outputs)
        outputs = self.second_dense_layer(outputs)
        return outputs, hiddens

    def step(self, batch, return_outputs=False):
        inputs, labels = batch
        outputs, hiddens = self(inputs)
        loss = F.cross_entropy(
            outputs.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=self.pad_token_id,
        )
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss, outputs = self.step(
            batch,
            return_outputs=True,
        )
        ppl = self.train_ppl(outputs, targets)
        self.log("ppl", ppl, prog_bar=True, on_step=True)
        self.log("loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        loss, outputs = self.step(
            batch,
            return_outputs=True,
        )
        ppl = self.train_ppl(outputs, targets)
        self.log("val_ppl", ppl, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        loss, outputs = self.step(
            batch,
            return_outputs=True,
        )
        ppl = self.train_ppl(outputs, targets)
        self.log("test_ppl", ppl)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #     )
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.75,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
# class PositionalEncoding(nn.Module):
#     r"""Inject some information about the relative or absolute position of the tokens in the sequence.
#         The positional encodings have the same dimension as the embeddings, so that the two can be summed.
#         Here, we use sine and cosine functions of different frequencies.
#     .. math:
#         \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#         \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#         \text{where pos is the word position and i is the embed idx)
#     Args:
#         d_model: the embed dim (required).
#         dropout: the dropout value (default=0.1).
#         max_len: the max. length of the incoming sequence (default=5000).
#     Examples:
#         >>> pos_encoder = PositionalEncoding(d_model)
#     """

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         Examples:
#             >>> output = pos_encoder(x)
#         """

#         x = x + self.pe[: x.size(0), :]
#         return self.dropout(x)


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


# Adding this model for further exploration and invistation
class LitTransformerLM(LightningModule):
    def __init__(
        self,
        vocab_size,
        heads=8,
        layers=3,
        dropout=0.1,
        pad_token_id=1,
        hidden_dim=2048,
        embeddings_dim=512,
        learning_rate=0.001,
    ):
        super(LitTransformerLM, self).__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.embed_dim = embeddings_dim
        self.pad_token_id = pad_token_id

        self.pos_encoder = PositionalEncoding(embeddings_dim, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            nhead=heads,
            dropout=dropout,
            batch_first=True,
            d_model=embeddings_dim,
            dim_feedforward=hidden_dim,
        )
        self.embedding = nn.Embedding(
            self.vocab_size,
            embeddings_dim,
            padding_idx=self.pad_token_id,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.linear = nn.Linear(embeddings_dim, self.vocab_size)
        # torch metrics
        self.train_ppl = torchmetrics.text.Perplexity(ignore_index=self.pad_token_id)
        self.val_ppl = torchmetrics.text.Perplexity(ignore_index=self.pad_token_id)
        self.test_ppl = torchmetrics.text.Perplexity(ignore_index=self.pad_token_id)

    def forward(self, src, **kwargs):
        src = self.embedding(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(
            self.device
        )
        output = self.transformer_encoder(src, mask=src_mask)
        output = self.linear(output)
        return output

    def step(self, batch, return_outputs=True):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(
            outputs.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=self.pad_token_id,
        )
        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss, outputs = self.step(batch)
        ppl = self.train_ppl(outputs, targets)
        self.log("ppl", ppl, prog_bar=True)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        loss, outputs = self.step(batch)
        ppl = self.train_ppl(outputs, targets)
        self.log("val_ppl", ppl, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        loss, outputs = self.step(batch)
        ppl = self.train_ppl(outputs, targets)
        self.log("test_ppl", ppl)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.learning_rate,
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
            "monitor": "val_loss",
        }
