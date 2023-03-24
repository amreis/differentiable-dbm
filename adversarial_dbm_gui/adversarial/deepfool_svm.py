from typing import Union
import numpy as np
import numpy.ma as ma
import torch as T
from sklearn.svm import LinearSVC
from tqdm import tqdm


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
    perturbed_batch = np.zeros_like(input_batch)
    perturbed_classes = np.zeros(input_batch.shape[0]) - 1
    i = 0
    for elem, cl in tqdm(zip(input_batch, orig_classes)):
        grad_diffs = weights - weights[cl, :]
        perturbed = elem.copy()
        r_hat = np.zeros_like(perturbed)
        for _ in range(max_iter):
            output = classifier.decision_function([perturbed]).squeeze()
            output_diffs = output - output[cl]
            output_diffs = ma.masked_array(
                output_diffs, mask=[int(i == cl) for i in np.unique(orig_classes)]
            )
            perturbations = np.abs(output_diffs) / np.linalg.norm(grad_diffs, axis=1)
            l_hat = np.argsort(perturbations)[0]

            r_i = (
                (perturbations[l_hat] + 1e-4)
                * grad_diffs[l_hat]
                / np.linalg.norm(grad_diffs[l_hat])
            )

            r_hat += r_i
            perturbed = elem + (1 + overshoot) * r_hat
            new_class = classifier.predict([perturbed]).item()
            if new_class != cl:
                perturbed_batch[i] = perturbed
                perturbed_classes[i] = new_class
        if perturbed_classes[i] == -1:
            perturbed_classes[i] = orig_classes[i]
            perturbed_batch[i] = input_batch[i]

        i += 1
    return perturbed_batch, perturbed_classes
