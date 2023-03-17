import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import logging

import numpy as np
import torch as T
import torch.nn as nn
from torch.utils.data import TensorDataset
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, minmax_scale
from scipy.spatial import distance

from .classifiers import nnclassifier
from .projection import nninv, qmetrics

DEVICE = "cuda" if T.cuda.is_available() else "cpu"

from numpy.typing import ArrayLike


class DataHolder:
    def __init__(self) -> None:
        self.X_classif_train: ArrayLike = None
        self.X_classif_test: ArrayLike = None
        self.y_classif_train: ArrayLike = None
        self.y_classif_test: ArrayLike = None
        self.X_tsne: ArrayLike = None
        self.X_proj_train: ArrayLike = None
        self.X_proj_test: ArrayLike = None
        self.X_high_train: ArrayLike = None
        self.X_high_test: ArrayLike = None
        self.classifier: nnclassifier.NNClassifier = None
        self.nninv_model: nninv.NNInv = None

    def _ndarray_fnames_to_attrs(self):
        return {
            "X_classif_train.npy": "X_classif_train",
            "X_classif_test.npy": "X_classif_test",
            "y_classif_train.npy": "y_classif_train",
            "y_classif_test.npy": "y_classif_test",
            "X_tsne.npy": "X_tsne",
            "X_proj_train.npy": "X_proj_train",
            "X_proj_test.npy": "X_proj_test",
            "X_high_train.npy": "X_high_train",
            "X_high_test.npy": "X_high_test",
        }

    def save_to_cache(self, path: str):
        if not self.check_all_data_present():
            logging.warn("some DataHolder attributes are still missing")
        ndarrays = self._ndarray_fnames_to_attrs()

        for fname, attr in ndarrays.items():
            np.save(os.path.join(path, fname), getattr(self, attr))

        T.save(self.classifier.state_dict(), os.path.join(path, "classifier.pth"))
        T.save(self.nninv_model.state_dict(), os.path.join(path, "nninv_model.pth"))

    def check_all_data_present(self) -> bool:
        return not any(
            getattr(self, attr) is None
            for attr in self._ndarray_fnames_to_attrs().values()
        )

    def load_from_cache(self, path: str):
        if not (os.path.exists(path) and os.path.isdir(path)):
            return False
        try:
            ndarrays = self._ndarray_fnames_to_attrs()

            for fname, attr in ndarrays.items():
                setattr(self, attr, np.load(os.path.join(path, fname)))

            self.classifier = nnclassifier.NNClassifier(
                self.X_classif_train.shape[1], len(np.unique(self.y_classif_train))
            )
            self.classifier.load_state_dict(
                T.load(os.path.join(path, "classifier.pth"))
            )
            self.classifier.to(device=DEVICE)
            self.nninv_model = nninv.NNInv(
                self.X_proj_train.shape[1], self.X_high_train.shape[1]
            )
            self.nninv_model.load_state_dict(
                T.load(os.path.join(path, "nninv_model.pth"))
            )
            self.nninv_model.to(device=DEVICE)

            return True
        except (FileNotFoundError, ValueError):
            return False


def train_nninv(X_proj, X_high, epochs=200, *, device: str = DEVICE) -> nninv.NNInv:
    model = nninv.NNInv(X_proj.shape[1], X_high.shape[1]).to(device=device)
    model.init_parameters()

    model.fit(
        TensorDataset(T.tensor(X_proj, device=device), T.tensor(X_high, device=device)),
        epochs=epochs,
        optim_kwargs={"lr": 1e-3},
    )
    return model


def train_classifier(
    X, y, n_classes: int, epochs: int = 50, *, device: str = DEVICE
) -> nnclassifier.NNClassifier:
    model = nnclassifier.NNClassifier(X.shape[1], n_classes=n_classes, act=nn.ReLU).to(
        device=device
    )
    model.init_parameters()

    model.fit(
        TensorDataset(T.tensor(X, device=device), T.tensor(y, device=device)),
        epochs=epochs,
    )

    return model


def read_and_prepare_data(dataset: str = "mnist", cache: bool = True) -> DataHolder:
    from .data import load_mnist

    holder = DataHolder()
    if os.path.exists(cache_path := os.path.join("data", "assets", dataset)):
        if holder.load_from_cache(cache_path):
            return holder

    X, y = load_mnist()

    X = minmax_scale(X).astype(np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    X_classif_train, X_classif_test, y_classif_train, y_classif_test = train_test_split(
        X, y, train_size=5000, test_size=1000, random_state=420, stratify=y
    )
    holder.X_classif_train = X_classif_train
    holder.X_classif_test = X_classif_test
    holder.y_classif_train = y_classif_train
    holder.y_classif_test = y_classif_test

    X_tsne = minmax_scale(
        TSNE(n_jobs=8, random_state=420).fit_transform(X_classif_train)
    ).astype(np.float32)
    holder.X_tsne = X_tsne

    D_high = distance.pdist(X_classif_train)
    D_low = distance.pdist(X_tsne)
    conts = qmetrics.per_point_continuity(D_high, D_low)

    keep_percent = 0.8
    c_keep_ixs = np.argsort(conts)[int((1 - keep_percent) * len(conts)) :]

    X_tsne_filtered = X_tsne[c_keep_ixs]

    X_proj_train, X_proj_test, X_high_train, X_high_test = train_test_split(
        X_tsne_filtered,
        X_classif_train[c_keep_ixs],
        train_size=1000,
        random_state=420,
        stratify=y_classif_train[c_keep_ixs],
    )
    holder.X_proj_train = X_proj_train
    holder.X_proj_test = X_proj_test
    holder.X_high_train = X_high_train
    holder.X_high_test = X_high_test

    classifier = train_classifier(X_classif_train, y_classif_train, n_classes)
    nninv_model = train_nninv(X_proj_train, X_high_train)

    holder.classifier = classifier
    holder.nninv_model = nninv_model

    if cache:
        base_path = os.path.join("data", "assets", dataset)
        holder.save_to_cache(base_path)
    return holder


from .components import plot, plotcontrols, datapoint
from .compute.dbm_manager import DBMManager


class MainWindow(tk.Frame):
    def __init__(self, root: tk.Tk, dbm_manager: DBMManager, *args, **kwargs) -> None:
        super().__init__(root, *args, **kwargs)
        self.root = root
        self.inverted = np.zeros((28, 28), dtype=np.float32)

        self.plot = plot.DBMPlot(self, dbm_manager)
        self.plot_controls = plotcontrols.Controls(self)
        self.plot_controls.attach(self.plot)
        self.inverted_vis = datapoint.DatapointFrame(self, self.inverted)

        self.plot.grid(column=0, row=0, rowspan=3)
        self.plot_controls.grid(column=1, row=0, sticky="NW")
        self.inverted_vis.grid(column=1, row=2, sticky="NSEW")

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)


def calculate(*args):
    print(args)


def main():
    import numpy as np

    from .components.plot import DBMPlot

    holder = read_and_prepare_data("mnist")

    root = tk.Tk()

    dbm_resolution = 200
    xx, yy = T.meshgrid(
        T.linspace(0, 1.0, dbm_resolution, device=DEVICE),
        T.linspace(0, 1.0, dbm_resolution, device=DEVICE),
        indexing="xy",
    )
    grid_points = T.stack([xx.ravel(), yy.ravel()], dim=1)

    dbm_manager = DBMManager(holder.classifier, holder.nninv_model, grid_points)
    # with T.no_grad():
    #     dbm = holder.classifier.classify(holder.nninv_model(grid_points))

    window = MainWindow(root, dbm_manager=dbm_manager)
    window.grid(column=0, row=0)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main()
