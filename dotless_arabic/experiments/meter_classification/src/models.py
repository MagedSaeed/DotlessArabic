import torchmetrics

import torch
from torch import nn
import torch.nn.functional as F

from dotless_arabic.experiments.meter_classification.src import constants

from pytorch_lightning import LightningModule


class LitMeterClassificationModel(LightningModule):
    def __init__(
        self,
        vocab_size,
        dropout_prob=0.333,
        number_of_classes=16,
        num_layers=constants.GRU_LAYERS,
        gru_hiddens=constants.HIDDEN_SIZE,
        gru_dropout=constants.GRU_DROPOUT,
        learning_rate=constants.LEARNING_RATE,
        embedding_size=constants.EMBEDDING_SIZE,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.gru_hiddens = gru_hiddens
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.number_of_classes = number_of_classes

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=number_of_classes,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=number_of_classes,
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=number_of_classes,
        )

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
        )
        self.gru_layer = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.gru_hiddens,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.first_dense_layer = nn.Linear(
            in_features=self.gru_hiddens,
            out_features=128,
        )
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()
        self.second_dense_layer = nn.Linear(
            in_features=128,
            out_features=self.number_of_classes,
        )

    def forward(self, x, hiddens=None):
        outputs = self.embedding_layer(x)
        outputs, hiddens = self.gru_layer(outputs)
        # https://stackoverflow.com/a/50914946/4412324
        outputs = (
            outputs[:, :, : self.gru_hiddens] + outputs[:, :, self.gru_hiddens :]
        )  # GRUs are bidirectional
        outputs = self.first_dense_layer(outputs)
        outputs = self.dropout_layer(outputs)
        outputs = self.relu(outputs)
        outputs = self.second_dense_layer(outputs)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        labels = labels.squeeze()  # drop unnecessary dimention
        outputs = outputs[:, -1, :]  # take the results at the last time-step
        loss = F.cross_entropy(outputs, labels)
        train_accuracy = self.train_accuracy(outputs, labels)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=False,
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
        outputs = self(inputs)
        labels = labels.squeeze()  # drop unnecessary dimention
        outputs = outputs[:, -1, :]  # take the results at the last time-step
        loss = F.cross_entropy(outputs, labels)
        val_accuracy = self.val_accuracy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_acc",
            val_accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        labels = labels.squeeze()  # drop unnecessary dimention
        outputs = outputs[:, -1, :]  # take the results at the last time-step
        preds = torch.argmax(
            F.softmax(outputs, dim=1),
            dim=1,
        )
        return labels, preds

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        labels = labels.squeeze()  # drop unnecessary dimention
        outputs = outputs[:, -1, :]  # take the results at the last time-step
        loss = F.cross_entropy(outputs, labels)
        test_accuracy = self.test_accuracy(outputs, labels)
        metrics = {"test_acc": test_accuracy, "test_loss": loss}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.1,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
