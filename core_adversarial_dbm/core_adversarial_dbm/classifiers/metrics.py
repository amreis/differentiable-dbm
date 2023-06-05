import torch as T
import torch.nn as nn

@T.no_grad()
def accuracy(classifier: nn.Module, X, y, device):
    outputs = classifier.classify(T.tensor(X, device=device))

    return (outputs.cpu().numpy() == y).mean()

