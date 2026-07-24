"""Literal implementation of the revised convolutional construction."""

import numpy as np


def _delta(x: np.ndarray, y: np.ndarray | int) -> np.ndarray:
    return np.maximum(0, 1 - np.abs(x - y))


def _causal_convolution(features: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    length = features.shape[1]
    padded = np.pad(features, ((0, 0), (length - 1, 0)))
    return np.stack(
        [padded[:, site : site + length] @ kernel for site in range(length)],
        axis=1,
    )


def evaluate(configurations: np.ndarray, colors: int) -> np.ndarray:
    symbols = configurations.astype(np.int64)
    increments = np.sign(symbols)
    color_labels = np.abs(symbols)
    length = symbols.shape[1]

    heights = _causal_convolution(increments, np.ones(length, dtype=np.int64))
    endpoint = np.zeros(length, dtype=np.int64)
    endpoint[-1] = 1
    height_gates = np.maximum(
        0,
        1
        - np.maximum(0, -heights)
        - endpoint[None, :] * np.maximum(0, heights),
    )

    if colors == 1:
        return np.prod(height_gates, axis=1)

    matched_colors = np.maximum(symbols, 0)
    for distance in range(2, length + 1):
        kernel = np.zeros(length, dtype=np.int64)
        kernel[length - distance] = 1
        candidate_color = _causal_convolution(color_labels, kernel)
        candidate_height = _causal_convolution(heights, kernel)
        candidate_increment = _causal_convolution(increments, kernel)
        matched_colors += (
            candidate_color
            * _delta(matched_colors, 0)
            * _delta(candidate_height, heights + 1)
            * _delta(candidate_increment, 1)
        )

    is_down = _delta(increments, -1)
    color_gates = 1 - is_down + _delta(matched_colors, color_labels) * is_down
    return np.prod(height_gates * color_gates, axis=1)
