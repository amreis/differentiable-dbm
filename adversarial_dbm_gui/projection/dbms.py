from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
from numpy.typing import ArrayLike


def find_wormholes(
    pixel_points: ArrayLike,
    pixel_classes: ArrayLike,
    adv_classes: ArrayLike,
    n_classes: int,
) -> tuple[ArrayLike, ArrayLike]:
    closest_mat = np.full((pixel_points.shape[0], n_classes), np.inf)
    for ix, point in enumerate(pixel_points):
        cl = pixel_classes[ix]

        for cl_j in range(n_classes):
            if cl == cl_j:
                continue

            closest = np.min(
                np.linalg.norm(
                    point[None, ...] - pixel_points[pixel_classes == cl_j], axis=1
                )
            )
            closest_mat[ix, cl_j] = closest
    closest_2d_class = np.argmin(closest_mat, axis=1)

    wormhole_dbm = np.where(closest_2d_class == adv_classes, pixel_classes, adv_classes)

    return wormhole_dbm, closest_2d_class


def wormholes(
    pixel_points: ArrayLike,
    pixel_classes: ArrayLike,
    adv_classes: ArrayLike,
    n_classes: int,
    resolution: int,
):
    assert resolution**2 == pixel_classes.shape[0]
    wormhole_dbm, _ = find_wormholes(
        pixel_points, pixel_classes, adv_classes, n_classes
    )

    return wormhole_dbm.reshape((resolution, resolution))


def dist_to_adv(
    inverted_grid_points: ArrayLike, closest_adv_points: ArrayLike, resolution: int
):
    assert resolution**2 == inverted_grid_points.shape[0]

    distances = np.linalg.norm(inverted_grid_points - closest_adv_points, axis=1)
    return distances.reshape((resolution, resolution))


def plot_heatmap(
    values: ArrayLike, *, ax: Optional[plt.Axes] = None, imshow_kwargs: dict = {}
):
    default_imshow_kwargs = {
        "cmap": "viridis",
        "extent": (0.0, 1.0, 0.0, 1.0),
        "interpolation": "none",
        "origin": "lower",
    }

    imshow_kwargs = default_imshow_kwargs | imshow_kwargs

    if ax is None:
        ax = plt.gcf().gca()
    return ax.imshow(values, **imshow_kwargs)


def plot_dbm(
    pixel_classes: ArrayLike, *, ax: Optional[plt.Axes] = None, imshow_kwargs: dict = {}
):
    default_imshow_kwargs = {
        "cmap": "tab10" if len(np.unique(pixel_classes)) <= 10 else "tab20",
        "extent": (0.0, 1.0, 0.0, 1.0),
        "interpolation": "none",
        "origin": "lower",
    }

    imshow_kwargs = default_imshow_kwargs | imshow_kwargs

    if ax is None:
        ax = plt.gcf().gca()
    return ax.imshow(pixel_classes, **imshow_kwargs)


def dbm_frontiers(pixel_classes: ArrayLike) -> ArrayLike:
    assert pixel_classes.ndim == 2

    nrows, ncols = pixel_classes.shape
    output = np.copy(pixel_classes).astype(np.float32)
    # First calculate edges horizontally
    for index, elem in np.ndenumerate(pixel_classes):
        row, col = index
        d_row, d_col = 0, 1

        in_bounds = (0 <= row + d_row < nrows) and (0 <= col + d_col < ncols)
        if not in_bounds:
            continue

        check = (row + d_row, col + d_col)
        neighbor = pixel_classes[check]

        if neighbor != elem:
            output[index] = -1

    # Now vertically, but using the output of the previous step as reference.
    # This means we won't create thick lines between regions, they should be
    # 1px thick =).
    for index, elem in np.ndenumerate(output):
        if elem == -1:
            continue

        row, col = index
        d_row, d_col = 1, 0

        in_bounds = (0 <= row + d_row < nrows) and (0 <= col + d_col < ncols)
        if not in_bounds:
            continue

        check = (row + d_row, col + d_col)
        neighbor = output[check]

        if -1 != neighbor != elem:
            output[index] = -1

    return np.where(output == -1, 0, np.nan)
