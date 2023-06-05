import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


class TargetedNNInv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def fit(self, dataset: Dataset, classifier, epochs: int, validation_data: Dataset, optim_kwargs: dict = {}):
        train_dl = DataLoader(dataset, batch_size=128, shuffle=True)
        recon_loss_fn = nn.MSELoss()
        classif_loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), **optim_kwargs)
        prev_val_loss = T.inf
        for e in range(epochs):
            epoch_loss = 0.0
            epoch_n = 0
            for batch in train_dl:
                self.zero_grad()

                projected, target = batch
                outputs = self(projected)
                loss = recon_loss_fn(outputs, target)

                loss += 0.1 * classif_loss_fn(classifier(outputs), classifier(target))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * target.size(0)
                epoch_n += target.size(0)
            if validation_data is not None:
                val_dl = DataLoader(validation_data, batch_size=128, shuffle=False)
                val_loss = 0.0
                val_n = 0
                with T.no_grad():
                    for val_batch in val_dl:
                        val_projected, val_target = val_batch
                        val_loss += recon_loss_fn(val_out := self(val_projected), val_target).item() * val_target.size(0)
                        val_loss += 0.1 * classif_loss_fn(classifier(val_out), classifier(val_target)).item() * val_target.size(0)
                        val_n += val_target.size(0)
                val_loss /= val_n
                if e > 50 and val_loss >= prev_val_loss:
                    print(f"Early stopping on epoch {e}. Val loss = {val_loss:.3f}")
                    break
                prev_val_loss = val_loss

            epoch_loss /= epoch_n
            if e % 50 == 0:
                print(f"Epoch {e} loss = {epoch_loss:.4f}")


def init_model(m: nn.Module):
    if isinstance(m, nn.Linear):
        # He Uniform initialization
        # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.constant_(m.bias, 0.01)

NNInv = TargetedNNInv