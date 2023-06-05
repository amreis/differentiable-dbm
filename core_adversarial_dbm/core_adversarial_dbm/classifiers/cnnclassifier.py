import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


class CNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: tuple[int, ...],
        n_classes: int,
        act: type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.n_classes = n_classes
        self._act = act

        self.conv1 = nn.Conv2d(input_dim[-1], 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 8, (3, 3))
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, n_classes)

    def activations(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.fc1(nn.Flatten()(x))
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, inputs):
        acts = self.activations(inputs)
        return F.softmax(acts, dim=-1)

    def classify(self, inputs):
        return T.max(self.forward(inputs), dim=1)[1]

    def prob_best_class(self, inputs):
        return T.max(self.forward(inputs), dim=1)[0]

    def init_parameters(self):
        self.apply(init_model)

    def fit(self, dataset: Dataset, epochs: int, optim_kwargs: dict = {}):
        train_dl = DataLoader(dataset, batch_size=256, shuffle=True)

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
            loop.set_description(f"Loss: {epoch_loss / epoch_n:.4f}")
        return self


def init_model(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
