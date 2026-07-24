"""Exhaustively verify the four revised Motzkin NQS constructions."""

import argparse

import numpy as np

import cnn
import fnn
import rnn
import transformer
from reference import enumerate_configurations, exact_indicator, motzkin_count


ARCHITECTURES = {
    "rnn": rnn,
    "fnn": fnn,
    "cnn": cnn,
    "transformer": transformer,
}


def verify_case(architecture: str, length: int, colors: int) -> bool:
    configurations = enumerate_configurations(length, colors)
    expected = exact_indicator(configurations)
    actual = ARCHITECTURES[architecture].evaluate(configurations, colors)
    mismatches = np.flatnonzero(actual != expected)
    passed = len(mismatches) == 0 and int(actual.sum()) == motzkin_count(
        length, colors
    )
    print(
        f"{'PASS' if passed else 'FAIL':4} {architecture:11} "
        f"s={colors} N={length} dimension={len(configurations)} "
        f"mismatches={len(mismatches)}"
    )
    if len(mismatches):
        index = int(mismatches[0])
        print(
            "     first mismatch:", configurations[index].tolist(),
            "actual=", actual[index], "expected=", expected[index],
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--colors", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    args = parser.parse_args()

    passed = sum(
        verify_case(architecture, length, colors)
        for architecture in args.architectures
        for colors in args.colors
        for length in args.sizes
    )
    total = len(args.architectures) * len(args.colors) * len(args.sizes)
    print(f"\n{passed}/{total} verification cases passed.")
    return int(passed != total)


if __name__ == "__main__":
    raise SystemExit(main())
