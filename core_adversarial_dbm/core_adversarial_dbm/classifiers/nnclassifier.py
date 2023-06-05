import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange


class NNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        layer_sizes: tuple[int, ...] = (512, 128, 32),
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.layers = [
            nn.Linear(in_features=i, out_features=o)
            for i, o in zip((input_dim,) + layer_sizes, layer_sizes + (n_classes,))
        ]
        self._act = act

        self.network = nn.Sequential()
        for layer in self.layers[:-1]:
            self.network.append(layer)
            self.network.append(self._act())
        self.network.append(self.layers[-1])

    def forward(self, inputs) -> T.Tensor:
        acts = self.activations(inputs)
        return F.softmax(acts, dim=-1)

    def activations(self, inputs) -> T.Tensor:
        return self.network(inputs)

    def classify(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[1].squeeze()

    def prob_best_class(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[0]

    def classification_entropy(self, inputs) -> T.Tensor:
        probs = self.forward(inputs)
        assert self.n_classes == probs.size(1)
        # probs = probs[probs > 0]
        entropy = T.where(probs > 0, -T.log(probs) * probs, 0.0)
        return (entropy / T.log(T.tensor(self.n_classes))).sum(dim=1)

    def init_parameters(self):
        self.apply(init_model)

    def fit(self, dataset: Dataset, epochs: int, optim_kwargs: dict = {}):
        train_dl = DataLoader(dataset, batch_size=128, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), **optim_kwargs)
        loop = trange(epochs)
        for e in loop:
            epoch_loss = 0.0
            epoch_n = 0

            for batch in train_dl:
                inputs, targets = batch

                self.zero_grad()
                outputs = self(inputs)

                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * targets.size(0)
                epoch_n += targets.size(0)
            loop.set_description(f"Loss: {epoch_loss/epoch_n:.4f}")
        return self


def init_model(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
