"""TODO: write NNInv code.

When we connect this to DeepFool we should be able to generate DBMs
that show distance to decision boundary in nD for every 2D point.
"""

from functools import partial

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class NNInv(nn.Module):
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
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.network(inputs)

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
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0.01)


def main():
    import matplotlib.pyplot as plt
    import torch.optim as optim
    from joblib import Memory
    from sklearn import datasets, model_selection, preprocessing
    from sklearn.manifold import TSNE

    from ..data import load_mnist

    device = "cuda" if T.cuda.is_available() else "cpu"

    X, y = load_mnist()
    X = preprocessing.minmax_scale(X).astype(np.float32)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    model = NNInv(2, X.shape[1]).to(device)
    model.apply(init_model)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=5000, test_size=2500, random_state=420, stratify=y
    )
    X_tsne = preprocessing.minmax_scale(TSNE().fit_transform(X_train))

    (
        X_train_inv,
        X_test_inv,
        X_tsne_train_inv,
        X_tsne_test_inv,
    ) = map(
        partial(T.tensor, device=device),
        model_selection.train_test_split(
            X_train,
            X_tsne,
            train_size=1000,
            test_size=4000,
            random_state=420,
            stratify=y_train,
        ),
    )

    model.fit(TensorDataset(X_tsne_train_inv, X_train_inv), epochs=200)

    test_dl = DataLoader(
        TensorDataset(X_tsne_test_inv, X_test_inv),
        batch_size=128,
    )

    total_mse = 0.0
    n = 0
    with T.no_grad():
        for batch in test_dl:
            projected, target = batch

            outputs = model(projected)

            total_mse += F.mse_loss(outputs, target) * target.size(0)
            n += target.size(0)
    print(f"Post-train Test MSE: {total_mse/n:.4f}")


if __name__ == "__main__":
    main()
