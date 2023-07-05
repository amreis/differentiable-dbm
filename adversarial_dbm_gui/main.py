import argparse
import logging
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch as T
import torch.nn as nn
from dotenv import load_dotenv
from MulticoreTSNE import MulticoreTSNE as TSNE
from numpy.typing import ArrayLike
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, minmax_scale
from torch.utils.data import TensorDataset
from umap import UMAP

from core_adversarial_dbm import defs
from core_adversarial_dbm.classifiers import metrics, nnclassifier
from core_adversarial_dbm.compute import neighbors
from core_adversarial_dbm.projection import nninv, nninv2, nninv_skip, qmetrics

DEVICE = defs.DEVICE


load_dotenv()


class DataHolder:
    def __init__(self) -> None:
        self.X_classif_train: ArrayLike = None
        self.X_classif_test: ArrayLike = None
        self.y_classif_train: ArrayLike = None
        self.y_classif_test: ArrayLike = None
        self.X_proj: ArrayLike = None
        self.X_proj_train: ArrayLike = None
        self.X_high_train: ArrayLike = None
        self.y_high_train: ArrayLike = None
        self.X_proj_val: ArrayLike = None
        self.X_high_val: ArrayLike = None
        self.classifier: nnclassifier.NNClassifier = None
        self.nninv_model: nninv.NNInv = None
        self.projection: str = None

    def _ndarray_fnames_to_attrs(self):
        return {
            "X_classif_train.npy": "X_classif_train",
            "X_classif_test.npy": "X_classif_test",
            "y_classif_train.npy": "y_classif_train",
            "y_classif_test.npy": "y_classif_test",
            "X_proj_train.npy": "X_proj_train",
            "X_high_train.npy": "X_high_train",
            "y_high_train.npy": "y_high_train",
            "X_proj_val.npy": "X_proj_val",
            "X_high_val.npy": "X_high_val",
        }

    def save_to_cache(self, path: str):
        if not self.check_all_data_present():
            logging.warn("some DataHolder attributes are still missing")
        # Checks if parent directory is missing as well.
        if not os.path.exists(os.path.join(path, self.projection)):
            # os.makedirs(path)
            os.makedirs(os.path.join(path, self.projection))
        ndarrays = self._ndarray_fnames_to_attrs()

        for fname, attr in ndarrays.items():
            np.save(os.path.join(path, fname), getattr(self, attr))

        np.save(os.path.join(path, self.projection, "X_proj.npy"), self.X_proj)
        T.save(self.classifier.state_dict(), os.path.join(path, "classifier.pth"))
        T.save(
            self.nninv_model.state_dict(),
            os.path.join(path, self.projection, "nninv_model.pth"),
        )

    def check_all_data_present(self) -> bool:
        return not any(
            getattr(self, attr) is None
            for attr in self._ndarray_fnames_to_attrs().values()
        )

    def load_inverter_from_cache(self, path: str, projection: str):
        try:
            self.nninv_model = nninv.NNInv(
                self.X_proj_train.shape[1], self.X_high_train.shape[1]
            )
            self.nninv_model.load_state_dict(
                T.load(
                    os.path.join(path, projection, "nninv_model.pth"),
                    map_location=DEVICE,
                )
            )
            self.nninv_model.to(device=DEVICE)
            self.projection = projection
        except OSError:
            self.nninv_model = None
            return False
        return True

    def load_projection_from_cache(self, path: str, projection: str):
        try:
            self.X_proj = np.load(os.path.join(path, projection, "X_proj.npy"))
        except OSError:
            return False
        return True

    def load_classifier_from_cache(self, path: str):
        if not os.path.isdir(path):
            return False
        try:
            self.classifier = nnclassifier.NNClassifier(
                self.X_classif_train.shape[1], len(np.unique(self.y_classif_train))
            )
            self.classifier.load_state_dict(
                T.load(os.path.join(path, "classifier.pth"), map_location=DEVICE)
            )
            self.classifier.to(device=DEVICE)
        except OSError:
            return False

        return True

    def load_dataset_from_cache(self, path: str):
        if not os.path.isdir(path):
            return False

        try:
            ndarrays = self._ndarray_fnames_to_attrs()
            for fname, attr in ndarrays.items():
                setattr(self, attr, np.load(os.path.join(path, fname)))
        except OSError:
            return False
        return True

    def load_from_cache(self, path: str, projection: str):
        if not (os.path.exists(path) and os.path.isdir(path)):
            return False
        try:
            return all(
                [
                    self.load_dataset_from_cache(path),
                    self.load_classifier_from_cache(path),
                    self.load_projection_from_cache(path, projection),
                    self.load_inverter_from_cache(path, projection),
                ]
            )
        except (OSError, FileNotFoundError, ValueError):
            return False


def train_nninv2(
    X_proj,
    X_high,
    classifier,
    X_proj_val=None,
    X_high_val=None,
    epochs=200,
    *,
    device: str = DEVICE,
):
    model = nninv2.TargetedNNInv(X_proj.shape[1], X_high.shape[1]).to(device=device)
    model.init_parameters()

    model.fit(
        TensorDataset(T.tensor(X_proj, device=device), T.tensor(X_high, device=device)),
        classifier,
        epochs=epochs,
        validation_data=None
        if X_proj_val is None
        else TensorDataset(
            T.tensor(X_proj_val, device=device), T.tensor(X_high_val, device=device)
        ),
        optim_kwargs={"lr": 1e-3},
    )
    return model


def train_nninv(
    X_proj,
    X_high,
    X_proj_val=None,
    X_high_val=None,
    epochs=200,
    *,
    device: str = DEVICE,
) -> nninv.NNInv:
    model = nninv.NNInv(X_proj.shape[1], X_high.shape[1]).to(device=device)
    model.init_parameters()

    model.fit(
        TensorDataset(T.tensor(X_proj, device=device), T.tensor(X_high, device=device)),
        epochs=epochs,
        validation_data=None
        if X_proj_val is None
        else TensorDataset(
            T.tensor(X_proj_val, device=device), T.tensor(X_high_val, device=device)
        ),
        optim_kwargs={"lr": 1e-3},
    )
    return model


def train_skip_nninv(
    X_proj, X_high, epochs=200, *, device: str = DEVICE
) -> nninv_skip.NNInvSkip:
    model = nninv_skip.NNInvSkip(X_proj.shape[1], X_high.shape[1]).to(device=device)
    model.init_parameters()

    model.fit(
        TensorDataset(T.tensor(X_proj, device=device), T.tensor(X_high, device=device)),
        epochs=epochs,
        optim_kwargs={"lr": 1e-3, "weight_decay": 1e-5},
    )
    return model


def train_classifier(
    X, y, n_classes: int, epochs: int = 100, *, device: str = DEVICE
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


def read_and_prepare_data(
    dataset: str, projection: str, skip_cached: bool = False, cache: bool = True
) -> DataHolder:
    from core_adversarial_dbm.data import (
        load_cifar10,
        load_fashionmnist,
        load_mnist,
        load_quickdraw,
    )

    holder = DataHolder()
    missing_data = True
    missing_classif = True
    missing_proj = True
    missing_invert = True
    if not skip_cached and os.path.exists(
        cache_path := os.path.join(defs.ROOT_PATH, os.environ["DATA_DIR"], dataset)
    ):
        # if we're missing_data, we'll have to regenerate the dataset and therefore everything else
        # because everything depends on the dataset. If we're only missing the classifier (resp.
        # inverter) then we don't invalidate anything else, because they're isolated from each other.
        missing_data = not holder.load_dataset_from_cache(cache_path)
        missing_classif = missing_data or not holder.load_classifier_from_cache(
            cache_path
        )
        missing_proj = missing_data or not holder.load_projection_from_cache(
            cache_path, projection
        )
        missing_invert = (
            missing_data
            or missing_proj
            or not holder.load_inverter_from_cache(cache_path, projection)
        )

        if not any([missing_data, missing_classif, missing_proj, missing_invert]):
            return holder
    if not missing_data:
        n_classes = len(np.unique(holder.y_classif_train))
    if missing_data:
        if dataset == "mnist":
            X, y = load_mnist()
        elif dataset == "fashionmnist":
            X, y = load_fashionmnist()
        elif dataset == "cifar10":
            X, y = load_cifar10()
        elif dataset == "quickdraw":
            X, y = load_quickdraw()

        X = minmax_scale(X).astype(np.float32)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        n_classes = len(label_encoder.classes_)

        (
            X_classif_train,
            X_classif_test,
            y_classif_train,
            y_classif_test,
        ) = train_test_split(
            X, y, train_size=5000, test_size=1000, random_state=420, stratify=y
        )
        holder.X_classif_train = X_classif_train
        holder.X_classif_test = X_classif_test
        holder.y_classif_train = y_classif_train
        holder.y_classif_test = y_classif_test

    if missing_proj:
        if projection == "tsne":
            X_proj = TSNE(n_jobs=8, random_state=420).fit_transform(
                holder.X_classif_train
            )
        elif projection == "umap":
            X_proj = UMAP(random_state=420).fit_transform(holder.X_classif_train)
        elif projection == "isomap":
            from sklearn.manifold import Isomap

            X_proj = Isomap().fit_transform(holder.X_classif_train)
        X_proj = minmax_scale(X_proj).astype(np.float32)
        holder.X_proj = X_proj

    D_high = distance.pdist(holder.X_classif_train)
    D_low = distance.pdist(holder.X_proj)
    conts = qmetrics.per_point_continuity(D_high, D_low)

    keep_percent = 0.8
    c_keep_ixs = np.argsort(conts)[int((1 - keep_percent) * len(conts)) :]

    X_proj_filtered = holder.X_proj[c_keep_ixs]

    holder.X_proj_train = X_proj_filtered.copy()
    holder.X_high_train = holder.X_classif_train[c_keep_ixs].copy()
    holder.y_high_train = holder.y_classif_train[c_keep_ixs].copy()

    if missing_classif:
        classifier = train_classifier(
            holder.X_classif_train, holder.y_classif_train, n_classes
        )
        holder.classifier = classifier
    print(
        f"Train Acc: {metrics.accuracy(holder.classifier, holder.X_classif_train, holder.y_classif_train, device=DEVICE)}"
    )
    print(
        f"Test Acc: {metrics.accuracy(holder.classifier, holder.X_classif_test, holder.y_classif_test, device=DEVICE)}"
    )

    if missing_invert:
        # Using validation. Should we?
        (
            holder.X_proj_train,
            holder.X_proj_val,
            holder.X_high_train,
            holder.X_high_val,
            holder.y_high_train,
            _,
        ) = train_test_split(
            holder.X_proj_train,
            holder.X_high_train,
            holder.y_high_train,
            stratify=holder.y_high_train,
            train_size=0.8,
            random_state=420,
        )
        nninv_model = train_nninv(
            holder.X_proj_train,
            holder.X_high_train,
            holder.X_proj_val,
            holder.X_high_val,
        )
        holder.nninv_model = nninv_model
        holder.projection = projection

    if cache:
        base_path = os.path.join(defs.ROOT_PATH, os.environ["DATA_DIR"], dataset)
        holder.save_to_cache(base_path)
    return holder


from core_adversarial_dbm.compute.dbm_manager import DBMManager
from core_adversarial_dbm.compute.neighbors import Neighbors

from .components import datapoint, plot


class MainWindow(tk.Frame):
    def __init__(
        self,
        root: tk.Tk,
        dbm_manager: DBMManager,
        data: DataHolder,
        neighbors: Neighbors,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, *args, **kwargs)
        self.root = root
        self.inverted = np.zeros((28, 28), dtype=np.float32)

        self.plot = plot.DBMPlot(self, dbm_manager, data, neighbors)
        self.inverted_vis = datapoint.DatapointFrame(self, self.inverted)

        self.plot.grid(column=0, row=0, rowspan=3, sticky="NSEW")
        self.inverted_vis.grid(column=1, row=2, sticky="NSEW")

        self.save_image_btn = ttk.Button(self, text="Save Plot", command=self.save_file)
        self.save_image_btn.grid(column=0, row=3)

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1, minsize=50)

    def save_file(self):
        from tkinter.filedialog import asksaveasfilename

        fname = asksaveasfilename(
            confirmoverwrite=True,
            initialdir=".",
            initialfile="Pic.png",
            filetypes=[("PNG Image", "*.png")],
        )

        if not fname:
            return

        self.plot.ax.set_axis_off()
        old_xlim = self.plot.ax.get_xlim()
        old_ylim = self.plot.ax.get_ylim()
        self.plot.ax.set_xlim(0.0, 1.0)
        self.plot.ax.set_ylim(0.0, 1.0)
        self.plot.fig.savefig(fname, dpi=400, bbox_inches="tight", pad_inches=0.0)
        self.plot.ax.set_axis_on()
        self.plot.ax.set_xlim(old_xlim)
        self.plot.ax.set_ylim(old_ylim)


def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    parser = argparse.ArgumentParser("adversarial_dbm_gui")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=("mnist", "fashionmnist", "cifar10", "quickdraw"),
    )
    parser.add_argument(
        "-p",
        "--projection",
        "--proj",
        type=str,
        default="tsne",
        choices=("tsne", "umap", "isomap"),
    )
    parser.add_argument(
        "--use-cached", dest="use_cached", action="store_true", default=True
    )
    parser.add_argument("--no-use-cached", dest="use_cached", action="store_false")
    parser.add_argument("--cache", dest="cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="cache", action="store_false")

    args = parser.parse_args()
    print(args)

    dataset: str = args.dataset
    projection: str = args.projection
    cache: bool = args.cache
    use_cached: bool = args.use_cached

    holder = read_and_prepare_data(
        dataset, projection, skip_cached=not use_cached, cache=cache
    )
    dbm_resolution = 300
    xx, yy = T.meshgrid(
        T.linspace(0.0, 1.0, dbm_resolution, device=DEVICE),
        T.linspace(0.0, 1.0, dbm_resolution, device=DEVICE),
        indexing="xy",
    )
    grid_points = T.stack([xx.ravel(), yy.ravel()], dim=1)
    n_classes = len(np.unique(holder.y_classif_train))
    recon = F.mse_loss(
        holder.nninv_model(T.tensor(holder.X_proj_train, device=DEVICE)),
        T.tensor(holder.X_high_train, device=DEVICE),
    )

    cached_classes = holder.classifier.classify(holder.nninv_model(grid_points))

    print(f"NNInv MSE: {recon:.4f}")

    # holder.nninv_model = train_skip_nninv(
    #     holder.X_proj_train, holder.X_high_train, epochs=300
    # )
    # recon = F.mse_loss(
    #     holder.nninv_model(T.tensor(holder.X_proj_train, device=DEVICE)),
    #     T.tensor(holder.X_high_train, device=DEVICE),
    # )
    # print(f"New NNInv MSE: {recon:.4f}")

    root = tk.Tk()
    root.tk.call("source", "./theme/forest-light.tcl")
    style = ttk.Style()
    print(style.theme_names())
    ttk.Style().theme_use("forest-light")

    dbm_manager = DBMManager(
        holder.classifier, holder.nninv_model, grid_points, n_classes
    )
    neighbors_db = neighbors.Neighbors(
        holder.nninv_model,
        holder.classifier,
        holder.X_high_train,
        holder.y_high_train,
        grid_points,
    )

    window = MainWindow(
        root, dbm_manager=dbm_manager, data=holder, neighbors=neighbors_db
    )
    window.grid(column=0, row=0, sticky=tk.NSEW)

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # plt.figure()
    # plt.imshow(
    #     cached_classes.detach().cpu().reshape((dbm_resolution, dbm_resolution)),
    #     origin="lower",
    #     interpolation="none",
    #     extent=(0.0, 1.0, 0.0, 1.0),
    #     cmap="tab10",
    # )
    # plt.show()

    root.mainloop()


if __name__ == "__main__":
    main()
