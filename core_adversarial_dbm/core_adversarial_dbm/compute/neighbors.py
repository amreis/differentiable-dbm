from functools import cache

import numpy as np
import torch as T
import torch.nn as nn
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
        self.n_grid_points = self.grid.size(0)
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

    def add_points(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.concatenate([self.X_train, X], axis=0)
        self.y_train = np.concatenate([self.y_train, y], axis=0)
        self.global_neighbor_finder.fit(self.X_train)
        for cl in range(self.n_classes):
            self.per_class_neighbor_finder[cl].fit(self.X_train[self.y_train == cl])
        self.cache_clear()

    def cache_clear(self):
        self.get_distance_to_nearest_neighbor.cache_clear()
        self.get_distance_to_nearest_same_class_neighbor.cache_clear()
        self.get_distance_to_nearest_diff_class_neighbor.cache_clear()

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
        distances = np.zeros((self.n_grid_points,), dtype=np.float32)
        with T.no_grad():
            inverted_grid = self.inverter(self.grid)
            inverted_grid_classes = self.classifier.classify(inverted_grid)
            inverted_grid = inverted_grid.cpu().numpy()
            inverted_grid_classes = inverted_grid_classes.cpu().numpy()

        for cl in range(self.n_classes):
            mask = inverted_grid_classes == cl
            if not np.any(mask):
                continue
            dist, _ = self.per_class_neighbor_finder[cl].kneighbors(
                inverted_grid[inverted_grid_classes == cl],
                n_neighbors=1,
                return_distance=True,
            )
            dist = dist.squeeze()
            distances[mask] = dist

        return distances.reshape((self.grid_width, self.grid_width)).copy()

    @cache
    def get_distance_to_nearest_diff_class_neighbor(self):
        distances = np.full((self.n_grid_points,), np.inf, dtype=np.float32)
        with T.no_grad():
            inverted_grid = self.inverter(self.grid)
            inverted_grid_classes = self.classifier.classify(inverted_grid)

            inverted_grid = inverted_grid.cpu().numpy()
            inverted_grid_classes = inverted_grid_classes.cpu().numpy()

        for cl in range(self.n_classes):
            mask = inverted_grid_classes == cl
            if not np.any(mask):
                continue
            elems = inverted_grid[inverted_grid_classes == cl]
            for other_cl in range(self.n_classes):
                if cl == other_cl:
                    continue

                dist, _ = self.per_class_neighbor_finder[other_cl].kneighbors(
                    elems, n_neighbors=1, return_distance=True
                )
                dist = dist.squeeze()
                distances[mask] = np.minimum(distances[mask], dist)
        return distances.reshape((self.grid_width, self.grid_width)).copy()
