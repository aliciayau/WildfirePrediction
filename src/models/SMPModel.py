from typing import Any
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from .BaseModel import BaseModel


class SMPModel(BaseModel):
    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            loss_function=loss_function,
            *args,
            **kwargs
        )
        self.save_hyperparameters()

        # Initialize U-Net model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels * (5 if flatten_temporal_dimension else 1),  # Account for temporal dimension
            classes=1,
        )

        # Define the loss function
        if loss_function == "BCE":
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_class_weight))
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    def forward(self, x):
        if self.hparams.flatten_temporal_dimension:
            # Flatten temporal dimension into the channel dimension
            batch_size, time_steps, channels, height, width = x.shape
            x = x.view(batch_size, time_steps * channels, height, width)
        else:
            # Use only the last observation
            x = x[:, -1, ...]

        return self.model(x)
