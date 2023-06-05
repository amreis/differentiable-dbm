from functools import partial

import torch as T
import torch.nn as nn
from torch.func import jacrev, vmap
from tqdm import tqdm

from ..classifiers.nnclassifier import NNClassifier


def targeted_deepfool_batch(
    model: nn.Module, input_batch: T.Tensor, target_class: int, max_iter: int = 50
):
    with T.no_grad():
        _, orig_classes = T.max(model(input_batch), dim=1)
        orig_classes = orig_classes.flatten()
    perturbed_points = input_batch.clone().detach()
    r_hat = T.zeros_like(perturbed_points)
    perturbed_points_final = [None for _ in range(input_batch.size(0))]
    perturbed_classes = orig_classes.clone().detach()
    q = [i for i in range(input_batch.size(0)) if orig_classes[i] != target_class]
    for _ in tqdm(range(max_iter)):
        if len(q) == 0:
            break
        jacobians = vmap(jacrev(model))(perturbed_points[q])
        outputs = model(perturbed_points[q])

        for index, output, jacobian in zip(q, outputs, jacobians):
            output_diff = output[target_class] - output[orig_classes[index]]
            grad_diff = jacobian[target_class] - jacobian[orig_classes[index]]
            perturbation = T.abs(output_diff) / T.norm(grad_diff, dim=1)
            r_i = (perturbation + 1e-4) * grad_diff
            r_hat[index] += r_i
            perturbed_points[index] = input_batch[index] + (1.02 * r_hat[index])
            perturbed_points[index].clip_(0.0, 1.0)
        _, new_classes = T.max(model(perturbed_points[q]), dim=1)
        to_remove_from_q = []
        for i, index in enumerate(q):
            if new_classes[i] != orig_classes[index]:
                perturbed_points_final[index] = perturbed_points[index]
                perturbed_classes[index] = new_classes[i]
                to_remove_from_q.append(i)
        # only works because to_remove_from_q is sorted by construction.
        q2 = list(q)
        for i in reversed(to_remove_from_q):
            q2.remove(q[i])
        q = q2
    for ix in [i for i, v in enumerate(perturbed_points_final) if v is None]:
        perturbed_points_final[ix] = perturbed_points[ix]
        perturbed_classes[ix] = orig_classes[ix]
    return T.vstack(perturbed_points_final), orig_classes, perturbed_classes


def _model_activations(model, inputs):
    acts = model.activations(inputs)
    return acts, acts


# Looks ugly because of all the [...[None]][0] workarounds to work with vmap().
# It's fast though.
def _calc_r_i(output, jacobian, orig_class):
    grad_diffs = jacobian - jacobian[orig_class[None]][0]
    output_diffs = output - output[orig_class[None]][0].unsqueeze(-1)
    perturbations = T.abs(output_diffs) / T.norm(grad_diffs, dim=1)
    l_hat = T.argsort(perturbations)[0]

    r_i = (
        (perturbations[l_hat[None]][0] + 1e-4)
        * grad_diffs[l_hat[None]][0]
        / T.norm(grad_diffs[l_hat[None]][0])
    )
    return r_i


def deepfool_batch(
    model: nn.Module, input_batch: T.Tensor, max_iter: int = 50, overshoot: float = 0.02
):
    with T.no_grad():
        _, orig_classes = T.max(model(input_batch), dim=1)
        orig_classes = orig_classes.flatten()
    perturbed_points = input_batch.clone().detach()
    r_hat = T.zeros_like(perturbed_points)
    perturbed_points_final = T.full_like(perturbed_points, T.nan)
    perturbed_classes = orig_classes.clone().detach()
    q = T.arange(input_batch.size(0), device=perturbed_points.device, dtype=T.long)
    loop = tqdm(range(max_iter))
    for i in loop:
        loop.set_description(f"{len(q) = }")
        if len(q) == 0:
            break
        jacobians, outputs = vmap(
            jacrev(partial(_model_activations, model), has_aux=True)
        )(perturbed_points[q])
        r_is = vmap(_calc_r_i)(outputs, jacobians, orig_classes[q])
        r_hat[q] += r_is
        perturbed_points[q] = input_batch[q] + ((1 + overshoot) * r_hat[q])
        _, new_classes = T.max(model(perturbed_points[q]), dim=1)

        (changed_classes,) = T.where(new_classes != orig_classes[q])
        perturbed_classes[q[changed_classes]] = new_classes[changed_classes]
        perturbed_points_final[q[changed_classes]] = perturbed_points[
            q[changed_classes]
        ]

        q = q[T.where(new_classes == orig_classes[q])]

    mask = T.isnan(perturbed_points_final)
    perturbed_points_final[mask] = perturbed_points[mask]
    perturbed_classes[mask.any(dim=1)] = orig_classes[mask.any(dim=1)]
    return perturbed_points_final, orig_classes, perturbed_classes


def deepfool(model: nn.Module, input_example: T.Tensor, max_iter: int = 50):
    return deepfool_batch(model, input_example[None, ...], max_iter=max_iter)


def deepfool_minibatches(
    model: nn.Module,
    input_batch: T.Tensor,
    batch_size: int = 10_000,
    max_iter: int = 50,
):
    from torch.utils.data import DataLoader, TensorDataset

    minibatches = DataLoader(
        TensorDataset(input_batch), batch_size=batch_size, shuffle=False
    )

    all_perturbed_points = []
    all_orig_classes = []
    all_perturbed_classes = []
    for batch in minibatches:
        (to_perturb,) = batch
        perturbed, orig_classes, perturbed_classes = deepfool_batch(
            model, to_perturb, max_iter=max_iter
        )

        all_perturbed_points.append(perturbed)
        all_orig_classes.append(orig_classes)
        all_perturbed_classes.append(perturbed_classes)

    return (
        T.cat(all_perturbed_points, dim=0),
        T.cat(all_orig_classes, dim=0),
        T.cat(all_perturbed_classes, dim=0),
    )


def main():
    from functools import partial

    import matplotlib.pyplot as plt
    import numpy as np
    import torch.optim as optim
    from joblib import Memory
    from sklearn import datasets, model_selection, preprocessing
    from torch.utils.data import DataLoader, TensorDataset

    memory = Memory("./tmp")
    fetch_openml_cached = memory.cache(datasets.fetch_openml)

    device = "cuda" if T.cuda.is_available() else "cpu"
    X, y = fetch_openml_cached(
        "mnist_784",
        return_X_y=True,
        cache=True,
        as_frame=False,
    )
    X = preprocessing.minmax_scale(X).astype(np.float32)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    model = NNClassifier(X.shape[1], len(label_encoder.classes_)).to(device)
    model.init_parameters()
    # Build datasets

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=5000, test_size=2500, random_state=420, stratify=y
    )
    X_train, X_test, y_train, y_test = map(
        partial(T.tensor, device=device), (X_train, X_test, y_train, y_test)
    )

    train_dl = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True,
    )
    test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    # Train
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    for e in range(epochs):
        epoch_loss = 0.0
        n_examples = 0

        for batch in train_dl:
            inputs, labels = batch

            model.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_examples += labels.size(0)
        if e % 10 == 0:
            print(f"Epoch {e}: Loss = {epoch_loss/n_examples:.4f}")

    # Print accuracy (global + per-class)
    per_class_correct = {k: 0 for k in range(n_classes)}
    per_class_n = dict(per_class_correct)
    with T.no_grad():
        correct = 0
        n_examples = 0
        for test_batch in test_dl:
            inputs, targets = test_batch
            outputs = model(inputs)
            _, classified_as = T.max(outputs, dim=1)
            is_correct = classified_as == targets
            for k in range(n_classes):
                per_class_n[k] += targets[targets == k].size(0)
                per_class_correct[k] += is_correct[targets == k].sum().item()

            correct += is_correct.sum()
            n_examples += targets.size(0)
        print(f"Model Accuracy: {100*(correct / n_examples):.2f}%")
        per_class_acc = {
            k: per_class_correct[k] / per_class_n[k] for k in range(n_classes)
        }
        print(f"Per-class model accuracy: {per_class_acc}")

    # DeepFool
    input_point = X_train[0]
    perturbed, orig_class, perturbed_class = deepfool_batch(
        model, input_point[None, ...]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)

    input_point_np = input_point.cpu().numpy()
    perturbed_np = perturbed.detach().cpu().numpy()
    ax1.imshow(input_point_np.reshape(28, 28), cmap="gray")
    ax2.imshow(perturbed_np.reshape(28, 28), cmap="gray")

    print(f"Distance in n-D space: {np.linalg.norm(perturbed_np - input_point_np):.4f}")
    plt.show()

    # DeepFool timing for 5000 elements.
    from time import perf_counter

    n_iter_acc = 0
    n_points = 0
    adv_class_counter = {k: [0 for _ in range(n_classes)] for k in range(n_classes)}
    start = perf_counter()
    orig_class_counter = [0 for _ in range(n_classes)]
    deepfool_dl = DataLoader(TensorDataset(X_train), shuffle=False, batch_size=256)
    for batch in deepfool_dl:
        (points,) = batch
        perturbed_points, orig_classes, perturbed_classes = deepfool_batch(
            model, points
        )
        for cl in orig_classes:
            orig_class_counter[cl] += 1
        for o, p in zip(orig_classes.cpu().numpy(), perturbed_classes.cpu().numpy()):
            adv_class_counter[o][p] += 1
        # for input_point, input_target in zip(points, targets):
        #     n_points += 1

        #     perturbed, n_iter, orig_class, perturbed_class = deepfool(
        #         model, input_point
        #     )
        #     n_iter_acc += n_iter
        #     adv_class_counter[orig_class.item()][perturbed_class.item()] += 1
    end = perf_counter()
    # print(f"Took on average {n_iter_acc / n_points} iterations to find adv. ex.")
    print(f"In total, the process took {end - start:.4f} seconds.")
    print(f"Orig classes: {orig_class_counter}")

    as_mat = [[adv_class_counter[row][col] for col in range(10)] for row in range(10)]
    print(as_mat)
    plt.imshow(as_mat, cmap="gray", origin="upper")
    plt.show()


if __name__ == "__main__":
    main()
