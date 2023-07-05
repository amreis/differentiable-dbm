import torch as T
import torch.nn as nn
import numpy as np

from core_adversarial_dbm.defs import DEVICE


@T.no_grad()
def accuracy(classifier: nn.Module, X, y, device=DEVICE):
    outputs = classifier.classify(T.tensor(X, device=device))

    return (outputs.cpu().numpy() == y).mean()


@T.no_grad()
def accuracy_per_class(classifier: nn.Module, X, y, device=DEVICE):
    outputs = classifier.classify(T.tensor(X, device=device)).cpu().numpy()

    classes = np.unique(y)
    accs = np.zeros(len(classes), dtype=np.float32)
    for cl in classes:
        accs[cl] = (outputs[y == cl] == cl).mean()
    return accs
