import torch as T
import numpy as np

from ..adversarial import deepfool


class DBMManager:
    def __init__(self, classifier, inverter, grid_points: T.Tensor) -> None:
        self.classifier = classifier
        self.inverter = inverter
        self.grid = grid_points
        self.dbm_resolution = int(np.sqrt(self.grid.shape[0]))  # might change later

        self._dbm_data = None
        self._wormhole_data = None
        self._dist_map = None
        self._conf_map = None

    def get_dbm_data(self):
        if self._dbm_data is None:
            with T.no_grad():
                self._dbm_data = (
                    self.classifier.classify(self.inverter(self.grid)).cpu().numpy()
                )
        return self._dbm_data.reshape((self.dbm_resolution, self.dbm_resolution))

    def get_wormhole_data(self):
        pass

    def get_distance_map(self):
        if self._dist_map is None:
            with T.no_grad():
                inverted_grid = self.inverter(self.grid)
                perturbed, orig_class, new_class = deepfool.deepfool_batch(
                    self.classifier, inverted_grid
                )
                self._dist_map = (
                    T.linalg.norm(inverted_grid - perturbed, dim=-1).cpu().numpy()
                )
        return self._dist_map.reshape((self.dbm_resolution, self.dbm_resolution))

    def get_conf_map(self):
        pass

    def invert_point(self, x: float, y: float):
        as_tensor = T.tensor([[x, y]], device=self.grid.device, dtype=T.float32)
        input_width = int(np.sqrt(self.classifier.input_dim))
        with T.no_grad():
            return (
                self.inverter(as_tensor)
                .cpu()
                .numpy()
                .reshape((input_width, input_width))
            )

    def reset_data(self):
        self._dbm_data = None
        self._wormhole_data = None
        self._dist_map = None
        self._conf_map = None
