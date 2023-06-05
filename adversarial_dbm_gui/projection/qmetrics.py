import numpy as np
import torch as T
from scipy.spatial.distance import pdist, squareform


def per_point_trustworthiness(D_high, D_low, k=7):
    D_high = squareform(D_high)
    D_low = squareform(D_low)

    n = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, : k + 1][:, 1:]
    knn_proj = nn_proj[:, : k + 1][:, 1:]

    trust_i = np.zeros(n, dtype=np.float32)
    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        for j in range(U.shape[0]):
            trust_i[i] += np.where(nn_orig[i] == U[j])[0] - k

    return (1 - (2 / (k * (2 * n - 3 * k - 1)) * trust_i)).squeeze()


def trustworthiness(D_high, D_low, k=7):
    return np.mean(per_point_trustworthiness(D_high, D_low, k=k)).squeeze()


def per_point_continuity(D_high, D_low, k=7):
    D_high = squareform(D_high)
    D_low = squareform(D_low)

    n = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, : k + 1][:, 1:]
    knn_proj = nn_proj[:, : k + 1][:, 1:]

    cont_i = np.zeros(n, dtype=np.float32)
    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        for j in range(V.shape[0]):
            cont_i[i] += np.where(nn_proj[i] == V[j])[0] - k

    return (1 - (2 / (k * (2 * n - 3 * k - 1)) * cont_i)).squeeze()


def per_point_jaccard(D_high, D_low, k=7):
    D_high = squareform(D_high)
    D_low = squareform(D_low)

    n = D_high.shape[0]

    knn_orig = D_high.argsort()[:, 1 : k + 1]
    knn_proj = D_low.argsort()[:, 1 : k + 1]

    jacc_i = np.zeros(n, dtype=np.float32)
    for i in range(n):
        I = np.intersect1d(knn_orig[i], knn_proj[i])
        U = np.union1d(knn_orig[i], knn_proj[i])

        jacc_i[i] = len(I) / len(U)

    return jacc_i


def interpolation(inverted_grid, ref_points):
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(ref_points)

    distances, _ = neighbors.kneighbors(
        inverted_grid, n_neighbors=1, return_distance=True
    )
    return np.mean(distances)


def surjectivity(inverted_grid, ref_points):
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(inverted_grid)

    distances, _ = neighbors.kneighbors(ref_points, n_neighbors=1, return_distance=True)
    return np.mean(distances)


def main():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, minmax_scale

    from ..data import load_mnist

    X, y = load_mnist()
    X = minmax_scale(X).astype(np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, random_state=420, stratify=y
    )

    from MulticoreTSNE import MulticoreTSNE as TSNE

    X_tsne = TSNE().fit_transform(X_train)

    import matplotlib.pyplot as plt

    D_high = pdist(X_train)
    D_low = pdist(X_tsne)

    trusts = per_point_trustworthiness(D_high, D_low, k=20)
    conts = per_point_continuity(D_high, D_low, k=20)

    keep_from_ix = int(0.2 * len(trusts))

    t_ixs = np.argsort(trusts)
    t_keep_ixs = t_ixs[keep_from_ix:]

    c_ixs = np.argsort(conts)
    c_keep_ixs = c_ixs[keep_from_ix:]

    j_ixs = np.argsort(per_point_jaccard(D_high, D_low, k=X_train.shape[0] // 10))
    j_keep_ixs = j_ixs[keep_from_ix:]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.scatter(*X_tsne.T, c=y_train)
    ax2.scatter(*X_tsne[t_keep_ixs].T, c=y_train[t_keep_ixs])
    ax3.scatter(*X_tsne[c_keep_ixs].T, c=y_train[c_keep_ixs])
    ax4.scatter(*X_tsne[j_keep_ixs].T, c=y_train[j_keep_ixs])
    plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
