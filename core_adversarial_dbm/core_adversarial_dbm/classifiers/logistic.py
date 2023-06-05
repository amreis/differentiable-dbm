import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import model_selection, preprocessing
from torch.utils.data import DataLoader, Dataset


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.W = nn.Parameter(T.zeros((n_classes - 1, input_dim), dtype=T.float32))
        self.bias = nn.Parameter(T.zeros((n_classes,), dtype=T.float32))

        self.linear = nn.Linear(
            in_features=self.input_dim, out_features=self.n_classes, bias=True
        )

    def forward(self, inputs):
        inputs = T.atleast_2d(inputs)
        predictions = inputs @ self.W.T

        expanded = T.cat(
            (predictions, T.zeros((inputs.size(0), 1), dtype=T.float32)), dim=1
        )
        return F.softmax(expanded + self.bias[None, ...], dim=-1).squeeze(dim=0)

    def fit(self, dataset: Dataset, epochs: int, optim_kwargs: dict = {}):
        train_dl = DataLoader(dataset, batch_size=128, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), **({"lr": 1e-3} | optim_kwargs))

        print_every = epochs // 10
        if print_every == 0:
            print_every = 1

        for e in range(epochs):
            epoch_loss = 0.0
            epoch_n = 0
            for batch in train_dl:
                inputs, targets = batch

                self.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_n += targets.size(0)

            if e % print_every == 0:
                print(f"Epoch {e}: Loss = {epoch_loss/epoch_n:.4f}")

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W)


def main():
    from sklearn.datasets import make_blobs

    from ..data import load_mnist

    X, y = load_mnist()
    X, y = make_blobs(n_samples=1200, n_features=10, centers=3)

    from sklearn.preprocessing import LabelEncoder, minmax_scale

    X = minmax_scale(X).astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    n_classes = len(encoder.classes_)
    l = LogisticRegression(X.shape[1], n_classes)
    l.init_parameters()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, random_state=420, stratify=y
    )

    from torch.utils.data import TensorDataset

    l.fit(TensorDataset(T.tensor(X_train), T.tensor(y_train)), epochs=150)

    with T.no_grad():
        _, outputs = T.max(l(T.tensor(X_test)), dim=1)
        correct = (outputs.cpu().numpy() == y_test).mean()
        print(f"Accuracy: {correct:.4f}")

    from MulticoreTSNE import MulticoreTSNE as TSNE

    from ..adversarial.deepfool import deepfool_batch

    X_tsne = minmax_scale(TSNE().fit_transform(X_train)).astype(np.float32)

    from scipy.spatial.distance import pdist

    from ..projection.qmetrics import per_point_continuity

    conts = per_point_continuity(pdist(X_train), pdist(X_tsne))
    keep_idxs = np.argsort(conts)[int(len(conts) * 0.2) :]
    X_tsne = X_tsne[keep_idxs]

    X_proj_train, X_proj_test, X_high_train, X_high_test = train_test_split(
        X_tsne,
        X_train[keep_idxs],
        train_size=0.1,
        random_state=420,
        stratify=y_train[keep_idxs],
    )
    from ..projection.nninv import NNInv

    inv = NNInv(X_proj_train.shape[1], X_high_train.shape[1])
    inv.init_parameters()
    inv.fit(
        TensorDataset(T.tensor(X_proj_train), T.tensor(X_high_train)),
        epochs=200,
        optim_kwargs={"lr": 1e-4},
    )

    dbm_resolution = 350
    xx, yy = np.meshgrid(
        np.linspace(0, 1.0, dbm_resolution),
        np.linspace(0, 1.0, dbm_resolution),
        indexing="xy",
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    with T.no_grad():
        inverted_grid = inv(T.tensor(grid_points))
        _, grid_classes = T.max(l(inverted_grid), dim=1)
        grid_classes = grid_classes.cpu().numpy()

        perturbed, orig_classes, new_classes = deepfool_batch(l, inverted_grid)
        distance_map = (
            T.linalg.norm(inverted_grid - perturbed, dim=-1).detach().cpu().numpy()
        )
    orig_classes = orig_classes.cpu().numpy()
    new_classes = new_classes.cpu().numpy()
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, subplot_kw={"aspect": 1})

    imshow_params = {
        "extent": (0.0, 1.0, 0.0, 1.0),
        "interpolation": "none",
        "origin": "lower",
    }
    ax1.scatter(*X_tsne.T, c=y_train[keep_idxs], cmap="tab10")
    ax2.imshow(
        grid_classes.astype(int).reshape(dbm_resolution, dbm_resolution),
        cmap="tab10",
        **imshow_params,
    )
    ax3.imshow(
        -distance_map.reshape(dbm_resolution, dbm_resolution),
        cmap="viridis",
        **imshow_params,
    )
    ax4.imshow(
        new_classes.reshape(dbm_resolution, dbm_resolution),
        cmap="tab10",
        **imshow_params,
    )
    fig.savefig("./images/3Blobs_10d.pdf", dpi=500)
    plt.show()

    # closest_mat = np.full((grid_points.shape[0], n_classes), np.inf)
    # from tqdm import tqdm

    # for ix, point in tqdm(enumerate(grid_points), total=grid_points.shape[0]):
    #     cl = grid_classes[ix]

    #     for cl_j in range(n_classes):
    #         if cl == cl_j:
    #             continue

    #         closest = np.min(
    #             np.linalg.norm(
    #                 point[None, ...] - grid_points[grid_classes == cl_j], axis=1
    #             )
    #         )
    #         closest_mat[ix, cl_j] = closest
    # closest_2d_class = np.argmin(closest_mat, axis=1)

    # dbm = np.where(closest_2d_class == new_classes, grid_classes, new_classes)

    # fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"aspect": 1})
    # ax1.imshow(
    #     grid_classes.astype(int).reshape(dbm_resolution, dbm_resolution),
    #     cmap="tab10",
    #     **imshow_params,
    # )
    # ax2.imshow(
    #     dbm.astype(int).reshape(dbm_resolution, dbm_resolution),
    #     cmap="tab10",
    #     **imshow_params,
    # )

    # plt.show()

    from .. import dbms

    frontiers = dbms.dbm_frontiers(grid_classes.reshape(dbm_resolution, dbm_resolution))
    wormholes = dbms.wormholes(
        grid_points, grid_classes.reshape(-1), new_classes, n_classes, dbm_resolution
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, subplot_kw={"aspect": 1}, figsize=(10, 5), dpi=500
    )

    dbms.plot_dbm(grid_classes.reshape(dbm_resolution, dbm_resolution), ax=ax1)
    dbms.plot_dbm(frontiers, ax=ax1, imshow_kwargs={"cmap": "gray"})
    dbms.plot_dbm(wormholes, ax=ax2, imshow_kwargs={"alpha": 1.0})
    dbms.plot_dbm(frontiers, ax=ax2, imshow_kwargs={"cmap": "gray"})

    fig.savefig("./images/Wormholes_3Blobs_10d.pdf")

    plt.close(fig)


if __name__ == "__main__":
    main()
