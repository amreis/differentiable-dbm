import argparse
import os
from typing import Callable, Union
from skimage.io import imread, imsave, ImageCollection
from skimage.util import montage
import pathlib
from glob import glob
from omegaconf import OmegaConf

from dataclasses import dataclass
from itertools import chain


@dataclass(frozen=True)
class Config:
    projection: str
    classifier: str
    dataset: str


def get_same_file_for_dirs(
    fname: str, base_dir: pathlib.Path, dirs: list[Union[pathlib.Path, str]]
) -> list[str]:
    return [(base_dir / proj_dir / fname).as_posix() for proj_dir in dirs]


def same_file_for_dirs_getter(
    base_dir: pathlib.Path, dirs: list[pathlib.Path]
) -> Callable[[str], list[str]]:
    return lambda x: get_same_file_for_dirs(x, base_dir, dirs)


def main():
    root_path = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="none")
    parser.add_argument("--projection", "-p", type=str)

    args = parser.parse_args()
    directory = pathlib.Path(root_path / args.dir)
    projection = args.projection

    assert directory.exists(), f"directory {directory.resolve()} does not exist"
    out_path = directory / projection
    out_path.mkdir(exist_ok=True)
    dir_per_config: dict[Config, str] = {}

    # Read all configs first
    configs = sorted(glob((directory / "*" / ".hydra" / "config.yaml").as_posix()))
    for conf in configs:
        conf_dict = OmegaConf.load(conf)
        dir_per_config[
            Config(
                projection=conf_dict.projection,
                classifier=conf_dict.classifier.name,
                dataset=conf_dict.dataset,
            )
        ] = pathlib.Path(conf).parent.parent.name

    # As a test, let's find all data pertaining to t-SNE.
    proj_configs = {
        k: v for k, v in dir_per_config.items() if k.projection == projection
    }
    proj_dirs = list(proj_configs.values())
    file_getter = same_file_for_dirs_getter(directory, proj_dirs)
    dataset_names = [conf.dataset for conf in proj_configs]
    n_datasets = len(dataset_names)
    dbms = ImageCollection(file_getter("DBM.png"))
    confidence = ImageCollection(file_getter("ConfidenceMap.png"))
    closest_any_train_point = ImageCollection(
        file_getter("ClosestAnyTrainPointDist.png")
    )
    closest_same_class_train_point = ImageCollection(
        file_getter("ClosestSameClassTrainPointDist.png")
    )
    closest_diff_class_train_point = ImageCollection(
        file_getter("ClosestDiffClassTrainPointDist.png")
    )
    for img_c in (
        dbms,
        confidence,
        closest_any_train_point,
        closest_same_class_train_point,
        closest_diff_class_train_point,
    ):
        save_image_collection(out_path, img_c)

    def remove_alpha(x):
        return x[..., :3]

    arr = list(
        chain.from_iterable(
            map(
                remove_alpha,
                [
                    dbms.concatenate(),
                    confidence.concatenate(),
                    closest_any_train_point.concatenate(),
                    closest_same_class_train_point.concatenate(),
                    closest_diff_class_train_point.concatenate(),
                ],
            )
        )
    )

    img = montage(
        arr,
        fill=[255, 255, 255],
        padding_width=20,
        grid_shape=(5, n_datasets),
        channel_axis=3,
    )
    imsave(out_path / "ConfidenceAndClosestTrainPoint.png", img)

    del (
        img,
        confidence,
        closest_any_train_point,
        closest_diff_class_train_point,
        closest_same_class_train_point,
    )

    # Get the per-class grad images. These are not composed across
    # datasets, so no `file_getter` used.

    for conf, dirname in proj_configs.items():
        ic = ImageCollection(
            (directory / dirname / "NormJacClassifWRTInvertedGrid_Cl_*.png").as_posix()
        )
        save_image_collection(
            out_path,
            ic,
            fname=f"NormJacClassifWRTInvertedGrid_{conf.dataset}_PerClass.png",
            grid_shape=(2, 5) if len(ic.files) == 10 else None,
        )
    grad_classif_wrt_inverted_grid = ImageCollection(
        file_getter("NormJacClassifWRTInvertedGrid.png")
    )
    for conf, dirname in proj_configs.items():
        ic = ImageCollection(
            (
                directory / dirname / "NormJacClassifAndInversionWRTGrid_Cl_*.png"
            ).as_posix()
        )
        save_image_collection(
            out_path,
            ic,
            fname=f"NormJacClassifAndInversionWRTGrid_{conf.dataset}_PerClass.png",
            grid_shape=(2, 5) if len(ic.files) == 10 else None,
        )
    grad_classif_inversion_wrt_grid = ImageCollection(
        file_getter("NormJacClassifAndInversionWRTGrid.png")
    )
    grad_pinv = ImageCollection(file_getter("NormJacInversionWRTGrid.png"))
    for img_c in (
        grad_classif_wrt_inverted_grid,
        grad_classif_inversion_wrt_grid,
        grad_pinv,
    ):
        save_image_collection(out_path, img_c)
    arr = list(
        chain.from_iterable(
            map(
                remove_alpha,
                [
                    dbms.concatenate(),
                    grad_pinv.concatenate(),
                    grad_classif_wrt_inverted_grid.concatenate(),
                    grad_classif_inversion_wrt_grid.concatenate(),
                ],
            )
        )
    )

    grad_img = montage(
        arr,
        fill=[255, 255, 255],
        padding_width=20,
        grid_shape=(4, n_datasets),
        channel_axis=3,
    )
    imsave(out_path / "GradientMaps.png", grad_img)

    del (
        grad_img,
        grad_pinv,
        grad_classif_inversion_wrt_grid,
        grad_classif_wrt_inverted_grid,
    )

    from itertools import product

    for dist_type, blend_type in product(
        ["", "Compl"],
        ["MultiplySaturation", "BlendSoftLight", "BlendOverlay", "BlendHSV"],
    ):
        img_colls = [
            ImageCollection(
                file_getter(
                    f"dist_adv/{dist_type}DistToAdv{blend_type}Power{power:.1f}.png"
                )
            )
            for power in (0.1, 0.5, 1.0, 1.5, 2.0)
        ]
        for img_c in img_colls:
            save_image_collection(out_path, img_c)
        arr = list(
            chain.from_iterable(
                map(
                    remove_alpha,
                    [dbms.concatenate()] + [img_c.concatenate() for img_c in img_colls],
                )
            )
        )
        img = montage(
            arr,
            fill=[255, 255, 255],
            padding_width=20,
            grid_shape=(6, n_datasets),
            channel_axis=3,
        )
        imsave(out_path / f"{dist_type}DistToAdv{blend_type}.png", img)

    # Single image collections

    for fname in (
        "Projected.png",
        "ShadedDBM_ClosestAnyTrainPointDist.png",
        "ShadedDBM_ClosestSameClassTrainPointDist.png",
        "ShadedDBM_ClosestDiffClassTrainPointDist.png",
        "ShadedDBM_NormJacClassifAndInversionWRTGrid.png",
        "ShadedDBM_NormJacClassifWRTInvertedGrid.png",
        "ShadedDBM_NormJacInversionWRTGrid.png",
        "ThresholdedDBM_ClosestAnyTrainPointDistP0.5.png",
        "ThresholdedDBM_ClosestDiffClassTrainPointDistP0.5.png",
        "ThresholdedDBM_ClosestSameClassTrainPointDistP0.5.png",
    ):
        save_image_collection(out_path, ImageCollection(file_getter(fname)))

    with open(out_path / "dataset_names.txt", "wt") as f:
        f.write("\n".join(dataset_names))
        f.write("\n")


def save_image_collection(
    out_path: pathlib.Path, img_c: ImageCollection, fname=None, grid_shape=None
):
    if fname is None:
        fname = pathlib.Path(img_c.files[0]).name

    n_cols = len(img_c.files)
    if grid_shape is None:
        grid_shape = (1, n_cols)

    img = montage(
        img_c.concatenate()[..., :3],
        fill=[255, 255, 255],
        padding_width=20,
        grid_shape=grid_shape,
        channel_axis=3,
    )
    imsave(out_path / fname, img)


if __name__ == "__main__":
    main()
