"""Exact Motzkin support obtained directly from the path definition."""

from itertools import product

import numpy as np


def enumerate_configurations(length: int, colors: int) -> np.ndarray:
    basis = (0, *range(1, colors + 1), *range(-1, -colors - 1, -1))
    return np.asarray(list(product(basis, repeat=length)), dtype=np.int16)


def exact_indicator(configurations: np.ndarray) -> np.ndarray:
    support = np.ones(len(configurations), dtype=np.int8)
    for row, configuration in enumerate(configurations):
        stack: list[int] = []
        for symbol in configuration:
            symbol = int(symbol)
            if symbol > 0:
                stack.append(symbol)
            elif symbol < 0:
                if not stack or stack.pop() != -symbol:
                    support[row] = 0
                    break
        if stack:
            support[row] = 0
    return support


def motzkin_count(length: int, colors: int) -> int:
    counts = [1] * (length + 1)
    for n in range(2, length + 1):
        counts[n] = counts[n - 1] + colors * sum(
            counts[k] * counts[n - 2 - k] for k in range(n - 1)
        )
    return counts[length]
