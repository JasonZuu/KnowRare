import torch
import torch.nn as nn


class AutoregressiveDecoder(nn.Module):
    def __init__(self, hidden_dims: int, ts_dims: int) -> None:
        """
        Decoder for autoregressive time series prediction.
        hidden_dims: Dimension of the encoded features
        ts_dims: Dimension of the time series features
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims, ts_dims)  # Predict the next step's features
        )

    def forward(self, encoded):
        """
        encoded: Output embedding from the LSTM encoder
        Returns the predicted next time step
        """
        return self.decoder(encoded)
