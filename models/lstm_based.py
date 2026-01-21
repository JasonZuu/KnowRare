import torch
import torch.nn as nn
import numpy as np

from utils.seed import set_random_seed


class LSTMBasedModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 demo_dims: int,
                 ts_dims: int,
                 hidden_dims: int = 128,
                 multi_cls: bool=False) -> None:
        """
            demo_dims: demographic features' dim
            ts_dims: time series features' dim
            num_classes: number of classes for classification
            hidden_dims: hidden dims for the model
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.multi_cls = multi_cls
        self.demo_encoder = nn.Sequential(nn.Linear(demo_dims, hidden_dims),
                                          nn.LeakyReLU(),
                                          nn.Linear(hidden_dims, hidden_dims),
                                          nn.LeakyReLU())
        self.ts_encoder = nn.LSTM(input_size=ts_dims,
                                    hidden_size=hidden_dims,
                                    batch_first=True)
        self.leaky_relu = nn.LeakyReLU()
        self.predictor = nn.Sequential(nn.Linear(hidden_dims*2, hidden_dims),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_dims, num_classes))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def _embed(self, demo, ts):
        demo = self.demo_encoder(demo)
        ts, _ = self.ts_encoder(ts)

        ts = ts[:, -1, :]  # get the last hidden state
        ts = self.leaky_relu(ts)
        total_emb = torch.concat((demo, ts), dim=-1)
        return total_emb

    def forward_with_hidden(self, demo, ts, use_output_activate: bool=True):
        self.ts_encoder.flatten_parameters()
        emb = self._embed(demo, ts)
        pred = self.predictor(emb)

        hidden = {"embed": emb,
                  "pred": pred}
        if use_output_activate:
            if self.multi_cls:
                y_score = self.softmax(pred)
            else:
                y_score = self.sigmoid(pred)
        else:
            y_score = pred
        return y_score, hidden

    def forward(self, demo, ts, use_output_activate: bool=True):
        pred, hidden = self.forward_with_hidden(demo, ts, use_output_activate=use_output_activate)
        return pred
    
    def load_embedor_state_dict(self, state_dict):
        """
        Loads the parameters of the model except for the predictor's parameters.

        Args:
            state_dict_path (str): Path to the state dictionary file.
        """
        # Filter out the parameters of the predictor
        filtered_state_dict = {key: value for key, value in state_dict.items() if not key.startswith("predictor")}
        self.load_state_dict(filtered_state_dict, strict=False)


def demo_lstm_based_model():
    set_random_seed(0)
    demo = torch.randn((64, 3))
    ts = torch.randn((64, 8, 42))

    model = LSTMBasedModel(demo_dims=3, ts_dims=42)
    pred = model(demo, ts)

    print(pred.shape)


if __name__ == "__main__":
    demo_lstm_based_model()

