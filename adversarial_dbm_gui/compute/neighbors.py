import numpy as np
import torch as T
import torch.nn as nn
from functools import cache

from sklearn.neighbors import NearestNeighbors


class Neighbors:
    def __init__(
        self,
        inverter: nn.Module,
        classifier: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        grid_points: T.Tensor,
    ) -> None:
        self.inverter = inverter
        self.classifier = classifier
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.grid = grid_points.clone()
        self.grid_width = int(np.sqrt(self.grid.shape[0]))
        self.global_neighbor_finder = NearestNeighbors(
            n_neighbors=5
        )  # TODO we only want 1 for now, though.
        self.global_neighbor_finder.fit(self.X_train)

        self.n_classes = len(np.unique(y_train))
        self.per_class_neighbor_finder = {}
        for cl in range(self.n_classes):
            neighbor_finder = NearestNeighbors(n_neighbors=5)
            neighbor_finder.fit(self.X_train[self.y_train == cl])
            self.per_class_neighbor_finder[cl] = neighbor_finder

    @cache
    def get_distance_to_nearest_neighbor(self):
        with T.no_grad():
            inverted_grid = self.inverter(self.grid).cpu().numpy()
        dist, _ = self.global_neighbor_finder.kneighbors(
            inverted_grid, n_neighbors=1, return_distance=True
        )
        return dist.reshape((self.grid_width, self.grid_width)).copy()

    @cache
    def get_distance_to_nearest_same_class_neighbor(self):
        # this retains the data type, which might be handy.
        distances = np.zeros_like(self.grid.cpu().numpy(), shape=(self.grid.shape[0],))
        with T.no_grad():
            inverted_grid = self.inverter(self.grid)
            inverted_grid_classes = self.classifier.classify(inverted_grid)
            inverted_grid = inverted_grid.cpu().numpy()
            inverted_grid_classes = inverted_grid_classes.cpu().numpy()

        for cl in range(self.n_classes):
            mask = inverted_grid_classes == cl
            dist, _ = self.per_class_neighbor_finder[cl].kneighbors(
                inverted_grid[inverted_grid_classes == cl],
                n_neighbors=1,
                return_distance=True,
            )
            dist = dist.squeeze()
            distances[mask] = dist

        return distances.reshape((self.grid_width, self.grid_width)).copy()

    def diff_between(self):  # TODO awful name. Change.
        return (
            self.get_distance_to_nearest_same_class_neighbor()
            - self.get_distance_to_nearest_neighbor()
        )
