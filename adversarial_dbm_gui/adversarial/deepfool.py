import torch as T
import torch.autograd.functional as tfunc
import torch.nn as nn
from functorch import jacrev, vmap

from ..classifiers.nnclassifier import NNClassifier


def deepfool_batch(model: nn.Module, input_batch: T.Tensor, max_iter: int = 50):
    with T.no_grad():
        _, orig_classes = T.max(model(input_batch), dim=1)
        orig_classes = orig_classes.flatten()
    perturbed_points = input_batch.clone().detach()
    r_hat = T.zeros_like(perturbed_points)
    perturbed_points_final = [None for _ in range(input_batch.size(0))]
    perturbed_classes = orig_classes.clone().detach()
    q = list(range(input_batch.size(0)))
    for _ in range(max_iter):
        if len(q) == 0:
            break
        jacobians = vmap(jacrev(model))(perturbed_points[q])
        # with T.no_grad():

        outputs = model(perturbed_points[q])
        for index, output, jacobian in zip(q, outputs, jacobians):
            grad_diffs = (jacobian - jacobian[orig_classes[index], :]).squeeze()
            output_diffs = (output - output[orig_classes[index], None]).squeeze()
            perturbations = T.abs(output_diffs) / T.norm(grad_diffs, dim=1)
            l_hat = T.argsort(perturbations)[0]

            r_i = (
                (perturbations[l_hat] + 1e-4)
                * grad_diffs[l_hat]
                # / T.norm(grad_diffs[l_hat])
            )
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
        for i in reversed(to_remove_from_q):
            del q[i]
    for ix in [i for i, v in enumerate(perturbed_points_final) if v is None]:
        perturbed_points_final[ix] = perturbed_points[ix]
        perturbed_classes[ix] = orig_classes[ix]
    return T.vstack(perturbed_points_final), orig_classes, perturbed_classes


# TODO FIX! This gives super huge perturbations. Use deepfool_batch instead.
def deepfool(model: nn.Module, input_example: T.Tensor, max_iter: int = 50):
    with T.no_grad():
        _, orig_class = T.max(model(input_example[None, ...]), dim=1)
        orig_class = orig_class.flatten()
    perturb = input_example.clone().detach()
    perturb_class = orig_class
    r_tot = T.zeros_like(perturb)
    for i in range(max_iter):
        if perturb_class != orig_class:
            break
        with T.no_grad():
            outputs = model(perturb[None, ...]).squeeze()
        J: T.Tensor = tfunc.jacobian(model, perturb[None, ...]).squeeze()

        grad_diffs = J - J[orig_class, :]
        output_diffs = (outputs - outputs[orig_class, None]).squeeze()
        # argsort over positive values will output the smallest one (0) in position 0
        # but we need to discard this because it's the distance of the class to itself,
        # so we look at index 1.
        l_hat = T.argsort(output_diffs.abs() / T.norm(grad_diffs, dim=1))[1]
        r = (
            T.abs(output_diffs[l_hat])
            * grad_diffs[l_hat]
            / T.norm(grad_diffs[l_hat], p=2) ** 2
        )

        r_tot = r_tot + r
        perturb += r
        result = T.max(model(perturb[None, ...]), dim=1)
        perturb_class = result[1].squeeze()

    return perturb, i, orig_class, perturb_class


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
    perturbed, orig_class, perturbed_class = deepfool_batch(model, input_point[None, ...])

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
