from concurrent import futures

import numpy as np
import torch as T

from core_adversarial_dbm.adversarial import deepfool
from core_adversarial_dbm.projection import dbms


class DBMManager:
    def __init__(
        self, classifier, inverter, grid_points: T.Tensor, n_classes: int
    ) -> None:
        self.classifier = classifier
        self.inverter = inverter
        self.grid = grid_points
        self.n_classes = n_classes
        self.dbm_resolution = int(np.sqrt(self.grid.shape[0]))  # might change later

        self._pool = futures.ThreadPoolExecutor(max_workers=2)
        self.reset_data()

        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.append(observer)

    def notify(self, event):
        for obs in self._observers:
            obs.update_params(event)

    def get_dbm_data(self, blocking=True):
        if blocking and self._dbm_data is None:
            self._dbm_data = (
                self.classifier.classify(self.inverter(self.grid)).cpu().numpy()
            )
            self._compute_all_adversarial_data()
            return self._dbm_data.reshape(
                (self.dbm_resolution, self.dbm_resolution)
            ).copy()
        if self._dbm_data is None:
            with T.no_grad():
                self._dbm_data = (
                    self.classifier.classify(self.inverter(self.grid)).cpu().numpy()
                )
            self._dist_map_fut = self._pool.submit(self._compute_distance_map)

        return self._dbm_data.reshape((self.dbm_resolution, self.dbm_resolution)).copy()

    def get_dbm_borders(self):
        if self._dbm_borders is None:
            self._dbm_borders = dbms.dbm_frontiers(self.get_dbm_data())
        return self._dbm_borders.copy()

    def _compute_wormhole_data(self):
        if self._naive_wormhole_data is None:
            self._compute_all_adversarial_data()

    def _compute_all_adversarial_data(self):
        with T.no_grad():
            inverted_grid = self.inverter(self.grid)
            (
                self._perturbed,
                self._orig_class,
                self._naive_wormhole_data,
            ) = (
                deepfool.deepfool_batch(self.classifier, inverted_grid)
                if inverted_grid.is_cuda
                else deepfool.deepfool_minibatches(
                    self.classifier, inverted_grid, batch_size=10000
                )
            )

            self._dist_map = (
                T.linalg.norm(inverted_grid - self._perturbed, dim=-1).cpu().numpy()
            )
            self._perturbed = self._perturbed.cpu().numpy()
            self._orig_class = self._orig_class.cpu().numpy()
            self._naive_wormhole_data = self._naive_wormhole_data.cpu().numpy()

    def get_naive_wormhole_data(self):
        if self._naive_wormhole_data is None:
            self._dist_map_fut.result() if self._dist_map_fut is not None else self._compute_all_adversarial_data()
        return self._naive_wormhole_data.reshape(
            self.dbm_resolution, self.dbm_resolution
        ).copy()

    def _find_frontiers(self, dbm):
        rows, cols = dbm.shape
        neighboring_class = {}
        for (i, j), cl in np.ndenumerate(dbm):
            for dx, dy in (
                (0, 1),  # prioritize 4-neighborhood
                (-1, 0),
                (0, -1),
                (1, 0),
                (1, 1),  # then check 8-neighborhood
                (-1, 1),
                (-1, -1),
                (1, -1),
            ):
                if not (0 <= dx + i < rows and 0 <= dy + j < cols):
                    continue
                if (neigh_cl := dbm[dx + i, dy + j]) != cl:
                    neighboring_class[(i, j)] = neigh_cl
                    break
        return neighboring_class

    def get_wormhole_data(self):
        if self._wormhole_data is None:
            if self._naive_wormhole_data is None:
                self.get_naive_wormhole_data()
            from collections import deque
            from itertools import product

            neighboring_class = self._find_frontiers(self.get_dbm_data())
            to_expand = deque(neighboring_class.keys())

            while to_expand:
                i, j = to_expand.popleft()
                for dx, dy in product([0, 1, -1], repeat=2):
                    if dx == dy == 0 or not (
                        0 <= dx + i < self.dbm_resolution
                        and 0 <= dy + j < self.dbm_resolution
                    ):
                        continue
                    new_point = (i + dx, j + dy)
                    if new_point in neighboring_class:
                        continue
                    neighboring_class[new_point] = neighboring_class[i, j]
                    to_expand.append(new_point)
            closest_2d_class = np.zeros_like(self.get_dbm_data())
            for point, cl in neighboring_class.items():
                closest_2d_class[point] = cl
            self._wormhole_data = np.where(
                closest_2d_class.reshape(-1) == self._naive_wormhole_data,
                self._orig_class,
                self._naive_wormhole_data,
            )
        return self._wormhole_data.reshape(
            (self.dbm_resolution, self.dbm_resolution)
        ).copy()

    def get_wormhole_data_nan(self):
        if self._wormhole_data is None:
            if self._naive_wormhole_data is None:
                self.get_naive_wormhole_data()
            from collections import deque
            from itertools import product

            neighboring_class = self._find_frontiers(self.get_dbm_data())
            to_expand = deque(neighboring_class.keys())

            while to_expand:
                i, j = to_expand.popleft()
                for dx, dy in product([0, 1, -1], repeat=2):
                    if dx == dy == 0 or not (
                        0 <= dx + i < self.dbm_resolution
                        and 0 <= dy + j < self.dbm_resolution
                    ):
                        continue
                    new_point = (i + dx, j + dy)
                    if new_point in neighboring_class:
                        continue
                    neighboring_class[new_point] = neighboring_class[i, j]
                    to_expand.append(new_point)
            closest_2d_class = np.zeros_like(self.get_dbm_data())
            for point, cl in neighboring_class.items():
                closest_2d_class[point] = cl
            self._wormhole_data = np.where(
                closest_2d_class.reshape(-1) == self._naive_wormhole_data,
                np.nan,
                self._naive_wormhole_data,
            )
        return self._wormhole_data.reshape(
            (self.dbm_resolution, self.dbm_resolution)
        ).copy()

    def _compute_distance_map(self, blocking=False):
        print("Computing distance map...")
        from time import perf_counter

        start = perf_counter()
        if self._dist_map is None:
            with T.no_grad():
                self._compute_all_adversarial_data()
        end = perf_counter()
        print("Distance map computed")
        print(f"Time taken: {end-start:.5f}s")

    def get_distance_map(self, blocking=False):
        if self._dist_map is None:
            if self._dist_map_fut is None:
                self._compute_distance_map()
            else:
                if not self._dist_map_fut.done():
                    return
                self._dist_map_fut.result()
        # self._dist_map_thread.result()
        return self._dist_map.reshape(self.dbm_resolution, self.dbm_resolution).copy()

    def get_conf_map(self):
        pass

    @T.no_grad()
    def invert_point(self, x: float, y: float):
        as_tensor = T.tensor([[x, y]], device=self.grid.device, dtype=T.float32)
        input_width = int(np.sqrt(self.classifier.input_dim))
        return (
            self.inverter(as_tensor).cpu().numpy().reshape((input_width, input_width))
        )

    def distance_to_adv_at(self, row, col):
        if self._dist_map is None:
            return None
        return self._dist_map[row * self.dbm_resolution + col]

    def distance_to_inverted_neighbors(self):
        with T.no_grad():
            inverted_grid = (
                self.inverter(self.grid)
                .cpu()
                .numpy()
                .reshape((self.dbm_resolution, self.dbm_resolution, 1))
            )
        from scipy.signal import oaconvolve

        kernel = (
            np.array(
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
            ).reshape((3, 3, 1))
            / 8.0
        )

        convolved = oaconvolve(inverted_grid, kernel, mode="same")
        return np.linalg.norm(convolved, axis=-1)

    def reset_data(self):
        self._dbm_data = None
        self._dbm_borders = None
        self._naive_wormhole_data = None
        self._wormhole_data = None
        self._dist_map = None
        self._dist_map_fut = None
        self._conf_map = None

    def destroy(self):
        print("Waiting on child threads to join...")
        self._pool.shutdown(wait=True, cancel_futures=True)
        # if self._dist_map_thread.is_alive():
        #     self._dist_map_thread.join()
