"""Literal implementation of the revised recurrent construction."""

import numpy as np


def evaluate(configurations: np.ndarray, colors: int) -> np.ndarray:
    support = np.ones(len(configurations), dtype=np.int8)

    for row, symbols in enumerate(configurations):
        height = 0
        height_support = 1
        color_support = 1
        stack: list[int] = []

        for site, symbol in enumerate(symbols):
            symbol = int(symbol)
            increment = int(np.sign(symbol))
            height += increment

            penalty = max(0, -height)
            if site == len(symbols) - 1:
                penalty += max(0, height)
            height_support *= max(0, 1 - penalty)

            if colors > 1 and symbol > 0:
                stack.append(symbol)
            elif colors > 1 and symbol < 0:
                matches = bool(stack) and stack[-1] == -symbol
                color_support *= int(matches)
                if matches:
                    stack.pop()

        support[row] = height_support * color_support

    return support
