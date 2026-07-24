"""Literal implementation of the revised feed-forward construction."""

import numpy as np


def _delta(x: np.ndarray, y: np.ndarray | int) -> np.ndarray:
    return np.maximum(0, 1 - np.abs(x - y))


def evaluate(configurations: np.ndarray, colors: int) -> np.ndarray:
    symbols = configurations.astype(np.int64)
    increments = np.sign(symbols)
    length = symbols.shape[1]

    prefix_matrix = np.tril(np.ones((length, length), dtype=np.int64))
    heights = increments @ prefix_matrix.T
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

    color_labels = np.abs(symbols)
    matched_colors = np.maximum(symbols, 0)
    for distance in range(2, length + 1):
        destination = np.arange(distance - 1, length)
        source = destination - distance + 1
        matched_colors[:, destination] += (
            color_labels[:, source]
            * _delta(matched_colors[:, destination], 0)
            * _delta(heights[:, source], heights[:, destination] + 1)
            * _delta(increments[:, source], 1)
        )

    is_down = _delta(increments, -1)
    color_gates = 1 - is_down + _delta(matched_colors, color_labels) * is_down
    return np.prod(height_gates * color_gates, axis=1)
