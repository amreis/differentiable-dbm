from joblib import Memory
from sklearn import datasets


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
