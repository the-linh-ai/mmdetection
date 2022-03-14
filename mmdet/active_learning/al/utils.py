from typing import List, Any
from collections import defaultdict

import numpy as np

from ..utils.dist_utils import gather_and_compare


def calculate_softmax_np(logits: np.ndarray, dim: int):
    """
    Numpy-version softmax.

    Args:
        logits (np.ndarray): Input logits.
        dim (int): Dimension along which softmax is calculated.
    """
    e_logits = np.exp(logits - np.max(logits, axis=dim, keepdims=True))
    return e_logits / np.sum(e_logits, axis=dim, keepdims=True)


def calculate_entropy_np(
    logits: np.ndarray,
    dim: int,
    normalized: bool,
    assert_normalized: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Numpy-version Shannon entropy calculation.

    Args:
        logits (np.ndarray): Input logits, normalized or not normalized.
        dim (int): Dimension along which entropy is calculated.
        normalized (bool): Whether `tensor` is normalized along `dim` axis.
            If not, a softmax layer will be applied to the input tensor.
        assert_normalized (bool): Whether to check if the array is normalized
            or not if `normalized` is True.
    """
    if not normalized:
        logits = calculate_softmax_np(logits, dim=dim)
    elif assert_normalized:
        logits_s = logits.sum(axis=dim)
        if not np.allclose(logits_s, 1.0):
            raise ValueError(
                "The array has not been normalized (e.g., softmaxed)"
            )
    logits = logits + eps
    entropy = - logits * np.log(logits)
    entropy = np.sum(entropy, axis=dim)
    return entropy


def calculate_bvsb_np(
    logits: np.ndarray,
    dim: int,
    normalized: bool,
    assert_normalized: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Calculate best-vs-second best values from a prediction map.

    Args:
        logits (np.ndarray): Input logits.
        dim (int): Dimension along which bvsb value is calculated.
        normalized (bool): Whether `tensor` is normalized along `dim` axis.
            If not, a softmax layer will be applied to the input tensor.
        assert_normalized (bool): Whether to check if the array is normalized
            or not if `normalized` is True.
    """
    if logits.shape[1] == 1:
        raise ValueError(
            f"Best-vs-second-best policy is not applicable for single-class "
            f"probabilities."
        )

    if not normalized:
        logits = calculate_softmax_np(logits, dim=dim)
    elif assert_normalized:
        logits_s = logits.sum(axis=dim)
        if not np.allclose(logits_s, 1.0):
            raise ValueError(
                "The array has not been normalized (e.g., softmaxed)"
            )

    bvsb_idxs = np.argpartition(
        -logits,
        kth=2,
        axis=dim,
    )

    bvsb_idxs_0 = np.take(bvsb_idxs, indices=[0], axis=dim)
    bvsb_0 = np.take_along_axis(logits, bvsb_idxs_0, axis=dim).squeeze(dim)
    bvsb_idxs_1 = np.take(bvsb_idxs, indices=[1], axis=dim)
    bvsb_1 = np.take_along_axis(logits, bvsb_idxs_1, axis=dim).squeeze(dim)

    bvsb = (bvsb_1 / bvsb_0)
    return bvsb


def get_unique_indices(
    x: List[Any],
    device: str,
    buffer_size: int = None,
):
    """
    Get indices of unique values in a given list. If a value has a duplicate
    in the list, only its first occurence is recorded. Elements in the list
    should be all hash-able.
    """
    # Use a dict to count occurences
    count = defaultdict(list)
    for i, obj in enumerate(x):
        count[obj].append(i)

    unique_indices = []
    for indices in count.values():
        unique_indices.append(indices[0])
    gather_and_compare(unique_indices, device, buffer_size=buffer_size)
    return unique_indices
