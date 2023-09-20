import torchmetrics

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule


class LitTopicModelingModel(LightningModule):
    def __init__(
        self,
        vocab_size,
        num_layers=2,
        num_classes=7,
        pad_token_id=1,
        rnn_hiddens=128,
        rnn_dropout=0.3,
        dropout_prob=0.45,
        embedding_size=256,
        learning_rate=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_token_id = pad_token_id

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.rnn_hiddens = rnn_hiddens
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
        )
        self.rnn = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.rnn_hiddens,
            num_layers=self.num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.dense_layer = nn.Linear(
            in_features=self.rnn_hiddens,
            out_features=num_classes,
        )
        self.softmax = nn.Softmax()

    def forward(self, x, hiddens=None):
        outputs = self.embedding_layer(x)
        outputs = self.dropout_layer(outputs)
        outputs, hiddens = self.rnn(outputs)
        outputs = self.dense_layer(outputs)
        outputs = self.softmax(outputs)
        return outputs

    def step(self, inputs, labels):
        outputs = self(inputs)
        outputs = outputs[:, -1, :]  # take the results at the last time-step
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.step(inputs, labels)
        loss = F.cross_entropy(outputs, labels)
        train_accuracy = self.train_accuracy(outputs, labels)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            train_accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.step(inputs, labels)
        loss = F.cross_entropy(outputs, labels)
        val_accuracy = self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.step(inputs, labels)
        loss = F.cross_entropy(outputs, labels)
        test_accuracy = self.test_accuracy(outputs, labels)
        metrics = {"test_acc": test_accuracy, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        # optimizer = torch.optim.RMSprop(
        #     self.parameters(),
        #     lr=self.learning_rate,
        # )
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
