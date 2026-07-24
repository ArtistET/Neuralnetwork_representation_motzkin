"""Literal finite-softmax implementation of the revised Transformer."""

import numpy as np


OMEGA = 100.0
BETA_FACTOR = 1
CHUNK_SIZE = 4096


def _causal_mask(length: int) -> np.ndarray:
    return np.where(
        np.arange(length)[None, :] <= np.arange(length)[:, None],
        0.0,
        -np.inf,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def _uniform_causal_attention(length: int) -> np.ndarray:
    return _softmax(_causal_mask(length))


def _colorless(symbols: np.ndarray) -> np.ndarray:
    length = symbols.shape[1]
    positions = np.arange(1, length + 1)
    attention = _uniform_causal_attention(length)
    prefix_averages = np.sign(symbols) @ attention.T
    heights = np.rint(prefix_averages * positions).astype(np.int64)
    endpoint = np.zeros(length, dtype=np.int64)
    endpoint[-1] = 1
    gates = np.maximum(
        0,
        1
        - np.maximum(0, -heights)
        - endpoint[None, :] * np.maximum(0, heights),
    )
    return np.prod(gates, axis=1).astype(np.int8)


def _colorful(symbols: np.ndarray) -> np.ndarray:
    length = symbols.shape[1]
    positions = np.arange(1, length + 1, dtype=np.float64)
    attention = _uniform_causal_attention(length)
    increments = np.sign(symbols)

    prefix_averages = increments @ attention.T
    heights = np.rint(prefix_averages * positions).astype(np.int64)
    previous_heights = heights - increments
    height_violations = np.maximum(0, -heights)

    beta = BETA_FACTOR * length * OMEGA
    queries = np.stack((heights, np.ones_like(heights)), axis=-1)
    keys = np.stack(
        (
            2.0 * beta * previous_heights,
            -beta * previous_heights**2 + OMEGA * positions[None, :],
        ),
        axis=-1,
    )
    logits = (
        np.einsum("btd,bid->bti", queries, keys)
        + _causal_mask(length)[None, :, :]
    )
    pointer_values = np.einsum("bti,bi->bt", _softmax(logits), symbols)
    color_violations = (symbols < 0) & (symbols + pointer_values != 0.0)

    height_average = height_violations @ attention[-1]
    color_average = color_violations @ attention[-1]
    y = np.maximum(0, 1 - np.abs(heights[:, -1]) - length * height_average)
    v = np.maximum(0, 1 - length * color_average)
    return np.rint(np.maximum(0, y + v - 1)).astype(np.int8)


def evaluate(configurations: np.ndarray, colors: int) -> np.ndarray:
    symbols = configurations.astype(np.int64)
    if colors == 1:
        return _colorless(symbols)
    return np.concatenate(
        [
            _colorful(symbols[start : start + CHUNK_SIZE])
            for start in range(0, len(symbols), CHUNK_SIZE)
        ]
    )
