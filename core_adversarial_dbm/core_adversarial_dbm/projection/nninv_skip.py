from functools import partial

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class NNInvSkip(nn.Module):
    """Module that is able to learn an inverse projection for
    any target projection method and dataset.

    See Espadoto, M., Rodrigues, F. C. M., Hirata, N. S. T., Hirata, R., Jr, & Telea, A. C. (2019).
    Deep Learning Inverse Multidimensional Projections.
    EuroVis Workshop on Visual Analytics (EuroVA). doi:10.2312/eurova.20191118
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_1 = nn.Linear(self.input_dim, 2048)
        self.layer_2 = nn.Linear(2048, 2048)
        self.layer_3 = nn.Linear(2048, 2048)
        self.layer_4 = nn.Linear(2048, 2048)
        self.layer_5 = nn.Linear(2048, self.output_dim)

    def forward(self, inputs):
        x_1 = self.layer_1(inputs)
        z_1 = F.tanh(x_1)

        x_2 = self.layer_2(z_1)
        z_2 = F.tanh(x_2)

        x_3 = self.layer_3(z_2)
        z_3 = F.tanh(x_3)

        x_4 = self.layer_4(z_3)
        z_4 = F.tanh(x_4)

        x_out = self.layer_5(z_4)
        return F.sigmoid(x_out)

    def init_parameters(self):
        self.apply(init_model)

    def fit(self, dataset: Dataset, epochs: int, optim_kwargs: dict = {}):
        train_dl = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
        )

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), **optim_kwargs)
        for e in range(epochs):
            epoch_loss = 0.0
            epoch_n = 0
            for batch in train_dl:
                self.zero_grad()
                projected, target = batch

                outputs = self(projected)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * target.size(0)
                epoch_n += target.size(0)
            epoch_loss /= epoch_n
            if e % 50 == 0:
                print(f"Epoch {e}: Loss = {epoch_loss:.4f}")


def init_model(m: nn.Module):
    if isinstance(m, nn.Linear):
        # He Uniform initialization
        # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.constant_(m.bias, 0.01)
