from typing import Union
import numpy as np
import torch as T
from sklearn.svm import LinearSVC


def deepfool_svm_batch(
    classifier: LinearSVC,
    input_batch: Union[T.Tensor, np.ndarray],
    max_iter: int = 50,
    overshoot: float = 0.02,
):
    if isinstance(input_batch, T.Tensor):
        input_batch = input_batch.clone().detach().cpu().numpy()

    orig_classes = classifier.predict(input_batch)

    weights = classifier.coef_  # shape: (n_classes, n_dims)

    perturbed_batch = input_batch.copy()
    perturbed_batch_final = np.zeros_like(perturbed_batch)
    r_hat = np.zeros_like(perturbed_batch)
    perturbed_classes = np.zeros(input_batch.shape[0]) - 1

    q = np.arange(input_batch.shape[0])
    for _ in range(max_iter):
        if len(q) == 0:
            break
        outputs = classifier.decision_function(perturbed_batch[q])
        # shape = (batch_size, n_classes)
        output_diffs = (
            outputs - outputs[range(len(outputs)), tuple(orig_classes[q]), None]
        )
        output_diffs[output_diffs == 0.0] = np.nan
        # shape = (batch_size, n_classes, n_dims)
        grad_diffs = weights[None, ...] - weights[orig_classes[q], None, ...]

        perturbations = np.abs(output_diffs) / np.linalg.norm(grad_diffs, axis=-1)
        l_hat = np.argsort(perturbations, axis=1)[:, 0]
        r_i = (
            (perturbations[range(len(q)), tuple(l_hat), None] + 1e-4)
            * grad_diffs[range(len(q)), tuple(l_hat), :]
            / np.linalg.norm(
                grad_diffs[range(len(q)), tuple(l_hat), :], axis=-1, keepdims=True
            )
        )
        r_hat[q] += r_i
        perturbed_batch[q] = input_batch[q] + (1 + overshoot) * r_hat[q]
        new_classes = classifier.predict(perturbed_batch[q])

        (changed_classes,) = np.where(new_classes != orig_classes[q])
        perturbed_classes[q[changed_classes]] = new_classes[changed_classes]
        perturbed_batch_final[q[changed_classes]] = perturbed_batch[q[changed_classes]]
        q = q[np.where(new_classes == orig_classes[q])]

    mask = np.isnan(perturbed_batch_final)
    perturbed_batch_final[mask] = perturbed_batch[mask]
    perturbed_classes[mask.any(axis=1)] = orig_classes[mask.any(axis=1)]

    return perturbed_batch_final, orig_classes, perturbed_classes
