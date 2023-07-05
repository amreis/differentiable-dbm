import numpy as np
from joblib import Memory
from sklearn import datasets
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from core_adversarial_dbm.defs import ROOT_PATH


def load_mnist():
    memory = Memory(ROOT_PATH / "tmp")
    fetch_openml_cached = memory.cache(datasets.fetch_openml)

    X, y = fetch_openml_cached(
        "mnist_784",
        return_X_y=True,
        cache=True,
        as_frame=False,
    )
    return X, y


def load_fashionmnist():
    memory = Memory(ROOT_PATH / "tmp")
    fetch_openml_cached = memory.cache(datasets.fetch_openml)

    X, y = fetch_openml_cached(
        "Fashion-MNIST", return_X_y=True, cache=True, as_frame=False
    )
    return X, y


def load_cifar10():
    cifar10 = CIFAR10("/tmp", download=True, transform=ToTensor())

    X = cifar10.data[..., 0]
    X = X.reshape(X.shape[0], -1)

    return X, cifar10.targets


def load_quickdraw():
    base_path = ROOT_PATH / "data" / "assets" / "quickdraw"
    X = np.load(base_path / "X.npy")
    y = np.load(base_path / "y.npy")
    return X, y
