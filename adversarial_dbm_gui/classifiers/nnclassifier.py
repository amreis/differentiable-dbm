import torch as T
import torch.nn as nn


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
        self.network.append(nn.Softmax(dim=-1))

    def forward(self, inputs) -> T.Tensor:
        return self.network(inputs)

    def classify(self, inputs) -> T.Tensor:
        return T.max(self.forward(inputs), dim=1)[1].squeeze()

    def init_parameters(self):
        self.apply(init_model)


def init_model(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
