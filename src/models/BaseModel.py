import pytorch_lightning as pl
from abc import ABC
import torch
import torch.nn as nn
import torchmetrics

class BaseModel(pl.LightningModule, ABC):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        loss_function: str,  # Explicitly accept loss_function
        *args,
        **kwargs
    ):
        super().__init__()  # Pass only relevant arguments to LightningModule
        
        # Assume temporal dimension is flattened if required
        temporal_features = n_channels * (5 if flatten_temporal_dimension else 1)  # Adjust this based on the time steps

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(temporal_features, 64, kernel_size=3, padding=1),  # First convolutional layer
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=1)  # Output layer with 1 channel
        )

        # Define the loss based on the passed loss_function
        if loss_function == "BCE":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Define metrics
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        # Flatten temporal dimension if necessary
        if len(x.shape) == 5:  # Shape: [batch, time_steps, channels, height, width]
            x = x.flatten(start_dim=1, end_dim=2)  # Combine time and channels dimensions

        return self.model(x).squeeze(1)  # Output shape: [batch, height, width]


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.train_f1(y_hat, y.int()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.float())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1(y_hat, y.int()), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
