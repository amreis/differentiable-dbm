from joblib import Memory
from sklearn import datasets

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Lambda

def load_mnist():
    memory = Memory("./tmp")
    fetch_openml_cached = memory.cache(datasets.fetch_openml)

    X, y = fetch_openml_cached(
        "mnist_784",
        return_X_y=True,
        cache=True,
        as_frame=False,
    )
    return X, y


def load_fashionmnist():
    memory = Memory("./tmp")
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
