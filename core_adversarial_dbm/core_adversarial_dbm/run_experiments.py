"""
In this code, we want to generate all the views for different
combinations of dataset, projection, and classifier. Save all
generated images, together with relevant metrics.
"""

import logging
import os
from typing import Literal

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
from dotenv import load_dotenv
from matplotlib.colors import LightSource, hsv_to_rgb, rgb_to_hsv
from MulticoreTSNE import MulticoreTSNE as TSNE
from numpy.typing import ArrayLike, NDArray
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, minmax_scale
from torch.utils.data import TensorDataset
from umap import UMAP

from core_adversarial_dbm import defs
from core_adversarial_dbm.classifiers import metrics, nnclassifier
from core_adversarial_dbm.compute import dbm_manager, gradient, neighbors
from core_adversarial_dbm.data import (
    load_cifar10,
    load_fashionmnist,
    load_mnist,
    load_quickdraw,
)
from core_adversarial_dbm.projection import nninv, nninv2, nninv_skip, qmetrics

DEVICE = "cuda" if T.cuda.is_available() else "cpu"


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


def setup_figure():
    fig, ax = plt.subplots(
        1, 1, subplot_kw={"aspect": "equal"}, figsize=(8, 8), dpi=256
    )
    ax.axis("off")
    ax.set_autoscale_on(False)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    return fig, ax


def shade_image_with(
    rgb,
    shading: NDArray,
    mode: Literal["multiply_sat", "hsl", "soft_light", "overlay"] = "hsl",
):
    ls = LightSource()

    if mode == "multiply_sat":
        hsv = rgb_to_hsv(rgb)
        shading = (shading - shading.min()) / (shading.max() - shading.min())
        hsv[..., 1] *= shading
        return hsv_to_rgb(hsv)
    elif mode == "hsl":
        return ls.blend_hsv(
            rgb,
            shading[..., None],
            hsv_min_sat=0.5,
            hsv_max_sat=1.0,
            hsv_min_val=0.3,
            hsv_max_val=1.0,
        )
    elif mode == "soft_light":
        blended = ls.blend_soft_light(rgb, shading[..., None])
        blended = (blended - blended.min()) / (blended.max() - blended.min())
        return blended
    elif mode == "overlay":
        blended = ls.blend_overlay(rgb, shading[..., None])
        blended = (blended - blended.min()) / (blended.max() - blended.min())
        return blended
    else:
        raise ValueError(f"invalid mode: {mode}")


class MapGenerator:
    def __init__(self, dataholder: DataHolder, dbm_resolution: int) -> None:
        self.holder = dataholder
        self.n_classes = len(np.unique(self.holder.y_classif_train))

        self.scalar_map_imshow_kwargs = {
            "extent": (0.0, 1.0, 0.0, 1.0),
            "interpolation": "none",
            "origin": "lower",
            "cmap": "viridis",
        }
        self.categorical_map_imshow_kwargs = {
            **self.scalar_map_imshow_kwargs,
            "vmin": 0,
            "vmax": self.n_classes - 1,
            "cmap": "tab10" if self.n_classes <= 10 else "tab20",
        }

        self.dbm_resolution = dbm_resolution
        self.dbm_shape = (self.dbm_resolution, self.dbm_resolution)

        xx, yy = T.meshgrid(
            T.linspace(0.0, 1.0, dbm_resolution, device=DEVICE),
            T.linspace(0.0, 1.0, dbm_resolution, device=DEVICE),
            indexing="xy",
        )
        self.grid_points = T.stack([xx.ravel(), yy.ravel()], dim=1)
        self.dbm_data = dbm_manager.DBMManager(
            self.holder.classifier,
            self.holder.nninv_model,
            self.grid_points,
            self.n_classes,
        )

        self.neighbor_maps = neighbors.Neighbors(
            self.holder.nninv_model,
            self.holder.classifier,
            self.holder.X_classif_train,
            self.holder.y_classif_train,
            self.grid_points,
        )

        self.grad_maps = gradient.GradientMaps(
            self.grid_points,
            self.holder.classifier.activations,
            self.holder.nninv_model,
        )

    def projection(self):
        fig, ax = setup_figure()
        ax.scatter(
            *self.holder.X_proj.T,
            c=self.holder.y_classif_train,
            cmap="tab10",
            vmin=0,
            vmax=self.n_classes - 1,
        )

        fig.savefig("Projected.png")
        plt.close(fig)
        np.save("Projected.npy", self.holder.X_proj)

        fig, ax = setup_figure()
        ax.scatter(
            *self.holder.X_proj.T,
            c=self.holder.y_classif_train,
            s=20,
            edgecolors="#FFFFFF",
            linewidths=0.3,
            vmin=0,
            vmax=self.n_classes - 1,
            cmap=self.categorical_map_imshow_kwargs["cmap"],
        )
        fig.savefig("Projected_Colored_Transparent.png", transparent=True)
        plt.close(fig)

        fig, ax = setup_figure()
        ax.scatter(
            *self.holder.X_proj.T,
            c="k",
            s=20,
        )
        fig.savefig("Projected_Black_Transparent.png", transparent=True)
        plt.close(fig)

    def dbm(self):
        dbm = self.dbm_data.get_dbm_data(blocking=True)
        fig, ax = setup_figure()
        ax.imshow(
            dbm,
            **self.categorical_map_imshow_kwargs,
        )
        fig.savefig("DBM.png")
        np.save("DBM.npy", dbm)
        plt.close(fig)

    @T.no_grad()
    def confidence(self):
        confidence_map = (
            self.holder.classifier.prob_best_class(
                self.holder.nninv_model(self.grid_points)
            )
            .cpu()
            .numpy()
        ).reshape(self.dbm_shape)
        fig, ax = setup_figure()
        ax.imshow(
            confidence_map,
            vmax=1.0,
            vmin=0.0,
            **self.scalar_map_imshow_kwargs,
        )
        fig.savefig("ConfidenceMap.png")
        np.save("ConfidenceMap.npy", confidence_map)
        plt.close(fig)

    @T.no_grad()
    def confidence_in_dbm_luminance(self):
        confidence_map = (
            self.holder.classifier.prob_best_class(
                self.holder.nninv_model(self.grid_points)
            )
            .cpu()
            .numpy()
        ).reshape(self.dbm_shape)
        dbm = self.dbm_data.get_dbm_data()
        fig0, ax0 = setup_figure()
        img = ax0.imshow(dbm, **self.categorical_map_imshow_kwargs)
        rgb = img.make_image(None, unsampled=True)[0][..., :3] / 255.0
        rows, cols, channels = rgb.shape
        plt.close(fig0)
        import colorsys
        from itertools import starmap

        hls = np.array(list(starmap(colorsys.rgb_to_hls, rgb.reshape((-1, 3)))))
        hls = hls.reshape((rows, cols, channels))

        hls[..., 1] *= confidence_map
        rgb_with_conf = np.array(
            list(starmap(colorsys.hls_to_rgb, hls.reshape((-1, 3))))
        ).reshape((rows, cols, channels))
        fig, ax = setup_figure()
        ax.imshow(
            rgb_with_conf,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation="none",
        )
        fig.savefig("DBMWithConfidenceInLuminance.png")
        plt.close(fig)

        fig, ax = setup_figure()
        ls = LightSource()
        blended = ls.blend_hsv(
            rgb,
            confidence_map[..., None],
            hsv_min_sat=0.5,
            hsv_max_sat=1.0,
            hsv_min_val=0.3,
            hsv_max_val=1.0,
        )
        ax.imshow(
            blended, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), interpolation="none"
        )
        fig.savefig("DBMWithConfidenceAsIllumination.png")
        plt.close(fig)

    @T.no_grad()
    def entropy(self):
        entropy_map = (
            self.holder.classifier.classification_entropy(
                self.holder.nninv_model(self.grid_points)
            )
            .cpu()
            .numpy()
        ).reshape(self.dbm_shape)
        fig, ax = setup_figure()
        ax.imshow(entropy_map, vmax=1.0, vmin=0.0, **self.scalar_map_imshow_kwargs)
        fig.savefig("EntropyMap.png")
        np.save("EntropyMap.npy", entropy_map)
        plt.close(fig)

        fig, ax = setup_figure()
        ax.imshow(
            entropy_map**0.1, vmin=0.0, vmax=1.0, **self.scalar_map_imshow_kwargs
        )
        fig.savefig("EntropyMapPower0.1.png")
        plt.close(fig)

    def nearest_training_point(self):
        closest_train_point = (
            self.neighbor_maps.get_distance_to_nearest_neighbor().reshape(
                self.dbm_shape
            )
        )
        fig, ax = setup_figure()
        ax.imshow(closest_train_point, vmin=0.0, **self.scalar_map_imshow_kwargs)
        fig.savefig("ClosestAnyTrainPointDist.png")
        np.save("ClosestAnyTrainPointDist.npy", closest_train_point)
        plt.close(fig)

    def nearest_same_class_training_point(self):
        closest_train_point_same_class = (
            self.neighbor_maps.get_distance_to_nearest_same_class_neighbor().reshape(
                self.dbm_shape
            )
        )
        fig, ax = setup_figure()
        ax.imshow(
            closest_train_point_same_class, vmin=0.0, **self.scalar_map_imshow_kwargs
        )
        fig.savefig("ClosestSameClassTrainPointDist.png")
        np.save("ClosestSameClassTrainPointDist.npy", closest_train_point_same_class)
        plt.close(fig)

    def nearest_diff_class_training_point(self):
        closest_train_point_diff_class = (
            self.neighbor_maps.get_distance_to_nearest_diff_class_neighbor().reshape(
                self.dbm_shape
            )
        )
        fig, ax = setup_figure()
        ax.imshow(
            closest_train_point_diff_class, vmin=0.0, **self.scalar_map_imshow_kwargs
        )
        fig.savefig("ClosestDiffClassTrainPointDist.png")
        np.save("ClosestDiffClassTrainPointDist.npy", closest_train_point_diff_class)
        plt.close(fig)

    def _raw_dist_to_adversarial(self, dist_map):
        fig, ax = setup_figure()
        ax.imshow(dist_map, vmin=0.0, **self.scalar_map_imshow_kwargs)
        fig.savefig("DistToAdvRaw.png")
        np.save("DistToAdvRaw.npy", dist_map)
        plt.close(fig)

    def _dist_to_adversarial_multiply_saturation(
        self, original_image_rgb, dist_map, power_transf=1.0
    ):
        hsv = rgb_to_hsv(original_image_rgb)
        hsv[..., 1] *= dist_map**power_transf
        return hsv_to_rgb(hsv)

    def distance_to_adversarial(self):
        import os

        os.makedirs("./dist_adv")
        os.chdir("./dist_adv")

        POWERS = (2.0, 1.5, 1.0, 0.5, 0.1)
        dist_to_adv = self.dbm_data.get_distance_map(blocking=True)
        self._raw_dist_to_adversarial(dist_to_adv)

        # linear-normalize distances for blending
        smallest, largest = dist_to_adv.min(), dist_to_adv.max()
        dist_to_adv = (dist_to_adv - smallest) / (largest - smallest)

        # Now the blended maps.
        # First we need a DBM plotted.
        fig0, ax0 = setup_figure()
        image = ax0.imshow(
            self.dbm_data.get_dbm_data(), **self.categorical_map_imshow_kwargs
        )

        rgba, *ignore = image.make_image(None, unsampled=True)
        rgb = rgba[..., :3] / 255.0
        plt.close(fig0)

        closer_desaturated = {
            p: self._dist_to_adversarial_multiply_saturation(
                rgb, dist_to_adv, power_transf=p
            )
            for p in POWERS
        }
        farther_desaturated = {
            p: self._dist_to_adversarial_multiply_saturation(
                rgb, 1 - dist_to_adv, power_transf=p
            )
            for p in POWERS
        }

        for power, image in closer_desaturated.items():
            fname = f"DistToAdvMultiplySaturationPower{power:.1f}.png"
            fig, ax = setup_figure()
            ax.imshow(
                image, interpolation="none", origin="lower", extent=(0.0, 1.0, 0.0, 1.0)
            )
            fig.savefig(fname)
            plt.close(fig)
        for power, image in farther_desaturated.items():
            fname = f"ComplDistToAdvMultiplySaturationPower{power:.1f}.png"
            fig, ax = setup_figure()
            ax.imshow(
                image, interpolation="none", origin="lower", extent=(0.0, 1.0, 0.0, 1.0)
            )
            fig.savefig(fname)
            plt.close(fig)

        # Now using Matplotlib blends
        self._dist_to_adversarial_blend_hsv(rgb, dist_to_adv, POWERS)
        self._dist_to_adversarial_blend_soft_light(rgb, dist_to_adv, POWERS)
        self._dist_to_adversarial_blend_overlay(rgb, dist_to_adv, POWERS)
        os.chdir("..")

    def _dist_to_adversarial_blend_hsv(self, rgb, dist_to_adv, powers):
        ls = LightSource()
        for power in powers:
            fname = f"DistToAdvBlendHSVPower{power:.1f}.png"
            blended = ls.blend_hsv(
                rgb,
                dist_to_adv[..., None] ** power,
                hsv_min_sat=0.5,
                hsv_max_sat=1.0,
                hsv_min_val=0.3,
                hsv_max_val=1.0,
            )
            fig, ax = setup_figure()
            ax.imshow(
                blended,
                interpolation="none",
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            fig.savefig(fname)
            plt.close(fig)

            fname = "Compl" + fname
            blended = ls.blend_hsv(
                rgb,
                (1 - dist_to_adv[..., None]) ** power,
                hsv_min_sat=0.5,
                hsv_max_sat=1.0,
                hsv_min_val=0.3,
                hsv_max_val=1.0,
            )
            fig, ax = setup_figure()
            ax.imshow(
                blended,
                interpolation="none",
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            fig.savefig(fname)
            plt.close(fig)

    def _dist_to_adversarial_blend_soft_light(self, rgb, dist_to_adv, powers):
        ls = LightSource()
        for power in powers:
            fname = f"DistToAdvBlendSoftLightPower{power:.1f}.png"
            blended = ls.blend_soft_light(rgb, dist_to_adv[..., None] ** power)
            blended = (blended - blended.min()) / (blended.max() - blended.min())
            fig, ax = setup_figure()
            ax.imshow(
                blended,
                interpolation="none",
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            plt.savefig(fname)
            plt.close(fig)

            fname = "Compl" + fname
            blended = ls.blend_soft_light(rgb, (1 - dist_to_adv[..., None]) ** power)
            blended = (blended - blended.min()) / (blended.max() - blended.min())
            fig, ax = setup_figure()
            ax.imshow(
                blended,
                interpolation="none",
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            plt.savefig(fname)
            plt.close(fig)

    def _dist_to_adversarial_blend_overlay(self, rgb, dist_to_adv, powers):
        ls = LightSource()
        for power in powers:
            fname = f"DistToAdvBlendOverlayPower{power:.1f}.png"
            blended = ls.blend_overlay(rgb, dist_to_adv[..., None] ** power)
            blended = (blended - blended.min()) / (blended.max() - blended.min())
            fig, ax = setup_figure()
            ax.imshow(
                blended,
                interpolation="none",
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            plt.savefig(fname)
            plt.close(fig)

            fname = "Compl" + fname
            blended = ls.blend_overlay(rgb, (1 - dist_to_adv[..., None]) ** power)
            blended = (blended - blended.min()) / (blended.max() - blended.min())
            fig, ax = setup_figure()
            ax.imshow(
                blended,
                interpolation="none",
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            plt.savefig(fname)
            plt.close(fig)

    @T.no_grad()
    def gradient_maps(self):
        grad = (
            self.grad_maps.norm_jac_classif_wrt_inverted_grid()
            .cpu()
            .reshape(self.dbm_shape)
        )
        fig, ax = setup_figure()
        ax.imshow(grad, **self.scalar_map_imshow_kwargs)
        fig.savefig("NormJacClassifWRTInvertedGrid.png")
        np.save("NormJacClassifWRTInvertedGrid.npy", grad)
        plt.close(fig)

        for cl in range(self.n_classes):
            fig, ax = setup_figure()
            grad = (
                self.grad_maps.norm_jac_classif_wrt_inverted_grid_for_class(cl)
                .reshape(self.dbm_shape)
                .cpu()
            )
            ax.imshow(grad, **self.scalar_map_imshow_kwargs)
            base_fname = f"NormJacClassifWRTInvertedGrid_Cl_{cl}"
            fig.savefig(f"{base_fname}.png")
            np.save(f"{base_fname}.npy", grad)
            plt.close(fig)

        grad = (
            self.grad_maps.norm_jac_classif_and_inversion_wrt_grid()
            .cpu()
            .reshape(self.dbm_shape)
        )
        fig, ax = setup_figure()
        ax.imshow(grad, **self.scalar_map_imshow_kwargs)
        fig.savefig("NormJacClassifAndInversionWRTGrid.png")
        np.save("NormJacClassifAndInversionWRTGrid.npy", grad)
        plt.close(fig)
        for cl in range(self.n_classes):
            fig, ax = setup_figure()
            grad = (
                self.grad_maps.norm_jac_classif_and_inversion_wrt_grid_for_class(cl)
                .reshape(self.dbm_shape)
                .cpu()
            )
            ax.imshow(grad, **self.scalar_map_imshow_kwargs)
            base_fname = f"NormJacClassifAndInversionWRTGrid_Cl_{cl}"
            fig.savefig(f"{base_fname}.png")
            np.save(f"{base_fname}.npy", grad)
            plt.close(fig)

        grad = (
            self.grad_maps.norm_jac_inversion_wrt_grid().cpu().reshape(self.dbm_shape)
        )
        fig, ax = setup_figure()
        ax.imshow(grad, **self.scalar_map_imshow_kwargs)
        fig.savefig("NormJacInversionWRTGrid.png")
        np.save("NormJacInversionWRTGrid.npy", grad)
        plt.close(fig)

    def _linear_normalize(self, data: NDArray):
        return (data - data.min()) / (data.max() - data.min())

    @T.no_grad()
    def gradient_maps_shading(self):
        fig0, ax0 = setup_figure()
        image = ax0.imshow(
            self.dbm_data.get_dbm_data(), **self.categorical_map_imshow_kwargs
        )
        rgba, *ignore = image.make_image(None, unsampled=True)
        rgb = rgba[..., :3] / 255.0
        del rgba
        plt.close(fig0)

        grad = self._linear_normalize(
            self.grad_maps.norm_jac_classif_wrt_inverted_grid()
            .reshape(self.dbm_shape)
            .cpu()
            .numpy()
        )

        shaded_dbm = shade_image_with(rgb, grad, mode="hsl")
        fig, ax = setup_figure()
        ax.imshow(
            shaded_dbm,
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.savefig("ShadedDBM_NormJacClassifWRTInvertedGrid.png")
        plt.close(fig)

        grad = self._linear_normalize(
            self.grad_maps.norm_jac_classif_and_inversion_wrt_grid()
            .reshape(self.dbm_shape)
            .cpu()
            .numpy()
        )
        shaded_dbm = shade_image_with(rgb, grad, mode="hsl")
        fig, ax = setup_figure()
        ax.imshow(
            shaded_dbm,
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.savefig("ShadedDBM_NormJacClassifAndInversionWRTGrid.png")
        plt.close(fig)

        grad = self._linear_normalize(
            self.grad_maps.norm_jac_inversion_wrt_grid()
            .reshape(self.dbm_shape)
            .cpu()
            .numpy()
        )
        shaded_dbm = shade_image_with(rgb, grad, mode="hsl")
        fig, ax = setup_figure()
        ax.imshow(
            shaded_dbm,
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.savefig("ShadedDBM_NormJacInversionWRTGrid.png")
        plt.close(fig)

    def _inverse_cdf(self, data, p=0.5, n_bins=20):
        counts, edges = np.histogram(data, bins=n_bins)

        cumulative = counts.astype(np.float32).cumsum()
        cumulative /= cumulative[-1]

        ix = np.searchsorted(cumulative, p)

        # Guarantees at least p*data_size elements.
        return edges[ix + 1]

    def closest_training_point_thresholded(self):
        dbm = self.dbm_data.get_dbm_data().astype(np.float32)
        closest_train_point = (
            self.neighbor_maps.get_distance_to_nearest_neighbor().reshape(
                self.dbm_shape
            )
        )
        thresh = self._inverse_cdf(closest_train_point.reshape(-1), p=0.5)

        fig, ax = setup_figure()
        dbm_0 = dbm.copy()
        dbm_0[np.where(closest_train_point > thresh)] = np.nan
        ax.imshow(dbm_0, **self.categorical_map_imshow_kwargs)
        fig.savefig("ThresholdedDBM_ClosestAnyTrainPointDistP0.5.png")
        plt.close(fig)

        closest_train_point = (
            self.neighbor_maps.get_distance_to_nearest_same_class_neighbor().reshape(
                self.dbm_shape
            )
        )
        thresh = self._inverse_cdf(closest_train_point.reshape(-1), p=0.5)
        fig, ax = setup_figure()
        dbm_0 = dbm.copy()
        dbm_0[np.where(closest_train_point > thresh)] = np.nan
        ax.imshow(dbm_0, **self.categorical_map_imshow_kwargs)
        fig.savefig("ThresholdedDBM_ClosestSameClassTrainPointDistP0.5.png")
        plt.close(fig)

        closest_train_point = (
            self.neighbor_maps.get_distance_to_nearest_diff_class_neighbor().reshape(
                self.dbm_shape
            )
        )
        thresh = self._inverse_cdf(closest_train_point.reshape(-1), p=0.5)
        fig, ax = setup_figure()
        dbm_0 = dbm.copy()
        dbm_0[np.where(closest_train_point > thresh)] = np.nan
        ax.imshow(dbm_0, **self.categorical_map_imshow_kwargs)
        fig.savefig("ThresholdedDBM_ClosestDiffClassTrainPointDistP0.5.png")
        plt.close(fig)

    def closest_training_point_shading(self):
        fig0, ax0 = setup_figure()
        image = ax0.imshow(
            self.dbm_data.get_dbm_data(), **self.categorical_map_imshow_kwargs
        )
        rgba, *ignore = image.make_image(None, unsampled=True)
        rgb = rgba[..., :3] / 255.0
        del rgba
        plt.close(fig0)

        closest_train_point = self._linear_normalize(
            self.neighbor_maps.get_distance_to_nearest_neighbor().reshape(
                self.dbm_shape
            )
        )
        shaded_dbm = shade_image_with(rgb, closest_train_point, mode="hsl")
        fig, ax = setup_figure()
        ax.imshow(
            shaded_dbm,
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.savefig("ShadedDBM_ClosestAnyTrainPointDist.png")
        plt.close(fig)

        closest_train_point = self._linear_normalize(
            self.neighbor_maps.get_distance_to_nearest_same_class_neighbor().reshape(
                self.dbm_shape
            )
        )
        shaded_dbm = shade_image_with(rgb, closest_train_point, mode="hsl")
        fig, ax = setup_figure()
        ax.imshow(
            shaded_dbm,
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.savefig("ShadedDBM_ClosestSameClassTrainPointDist.png")
        plt.close(fig)

        closest_train_point = self._linear_normalize(
            self.neighbor_maps.get_distance_to_nearest_diff_class_neighbor().reshape(
                self.dbm_shape
            )
        )
        shaded_dbm = shade_image_with(rgb, closest_train_point, mode="hsl")
        fig, ax = setup_figure()
        ax.imshow(
            shaded_dbm,
            interpolation="none",
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        fig.savefig("ShadedDBM_ClosestDiffClassTrainPointDist.png")
        plt.close(fig)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup common params for Matplotlib.
    plt.rcParams.update(
        {
            "savefig.dpi": 256,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.0,
            "lines.markersize": 2.0,
        }
    )

    # print(cfg)
    load_dotenv()

    map_gen = MapGenerator(
        read_and_prepare_data(dataset=cfg.dataset, projection=cfg.projection),
        cfg.dbm_resolution,
    )

    print(os.getcwd())

    map_gen.projection()

    map_gen.dbm()

    map_gen.confidence()
    map_gen.confidence_in_dbm_luminance()
    map_gen.entropy()

    map_gen.nearest_training_point()
    map_gen.nearest_same_class_training_point()
    map_gen.nearest_diff_class_training_point()

    map_gen.distance_to_adversarial()
    map_gen.gradient_maps()

    map_gen.closest_training_point_thresholded()
    map_gen.closest_training_point_shading()
    map_gen.gradient_maps_shading()

    print(f"Done for {cfg}")


if __name__ == "__main__":
    main()
