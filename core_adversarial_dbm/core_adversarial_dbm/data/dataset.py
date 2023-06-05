import os
import numpy as np
import torch as T
from adversarial_dbm_gui.classifiers import nnclassifier

from ..projection import nninv

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE

from umap import UMAP
from sklearn.manifold import MDS


from numpy.typing import NDArray


class DataBunch:
    def __init__(self, device: str) -> None:
        self.device = device

        self.X_classif_train: NDArray = None
        self.X_classif_test: NDArray = None
        self.y_classif_train: NDArray = None
        self.y_classif_test: NDArray = None
        self.X_proj: NDArray = None
        self.X_proj_train: NDArray = None
        self.X_high_train: NDArray = None
        self.y_high_train: NDArray = None
        self.classifier = None
        self.inverter: nninv.NNInv = None

    @staticmethod
    def _ndarray_fnames_to_attrs():
        return {
            "X_classif_train.npy": "X_classif_train",
            "X_classif_test.npy": "X_classif_test",
            "y_classif_train.npy": "y_classif_train",
            "y_classif_test.npy": "y_classif_test",
            "X_proj_train.npy": "X_proj_train",
            "X_high_train.npy": "X_high_train",
            "y_high_train.npy": "y_high_train",
        }

    def _dataset_from_cache(self, path: str):
        if not os.path.isdir(path):
            return False
        ndarrays = DataBunch._ndarray_fnames_to_attrs()
        for fname, attr in ndarrays.items():
            file_path = os.path.join(path, fname)
            setattr(self, attr, np.load(file_path))

    def load(self, dataset: str, projection: str, *, use_cache=True):
        base_path = os.path.join("data", "assets")
        data_path = os.path.join(base_path, dataset)
        if use_cache and not self._dataset_from_cache(data_path):
            # TODO manually load data.
            pass

        self.X_proj = np.load(os.path.join(data_path, projection, "X_proj.npy"))

        self.classifier = nnclassifier.NNClassifier(
            self.X_classif_train.shape[1], len(np.unique(self.y_classif_train))
        )
        self.classifier.load_state_dict(
            T.load(os.path.join(data_path, "classifier.pth"), map_location=self.device)
        )
        self.classifier.to(device=self.device)
        self.inverter = nninv.NNInv(
            self.X_proj_train.shape[1], self.X_high_train.shape[1]
        )
        self.inverter.load_state_dict(
            T.load(
                os.path.join(data_path, projection, "nninv_model.pth"),
                map_location=self.device,
            )
        )
        self.inverter.to(device=self.device)

    @staticmethod
    def load(dataset: str, projection: str) -> "DataBunch":
        pass


def try_load_from_cache(dataset: str, projection: str) -> DataBunch:
    bunch = DataBunch()
    bunch.load(dataset, projection)


def load_data_bunch(
    dataset: str, projection: str, *, use_cache: bool = True
) -> DataBunch:
    if use_cache:
        bunch = try_load_from_cache(dataset, projection)
