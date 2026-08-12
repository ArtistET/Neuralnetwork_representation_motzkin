#!/usr/bin/env python3
"""Exhaustively verify an exact feedforward network for N-queens.

For every binary N x N board, this program compares

1. an independent, classical N-queens legality test, and
2. the output of the fixed-weight feedforward construction.

No training or third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


Board = tuple[int, ...]
Matrix = tuple[tuple[int, ...], ...]


def build_weight_matrix(n: int) -> Matrix:
    """Return the (6N-2) x N^2 binary incidence matrix W.

    Its rows represent, in order, the N board rows, N board columns,
    2N-1 main diagonals (i-j constant), and 2N-1 anti-diagonals
    (i+j constant).  Board coordinates and flattened indices are zero-based:
    k = N*i + j.
    """
    if n < 1:
        raise ValueError("N must be positive")

    lines: list[list[int]] = []

    for i in range(n):
        lines.append([n * i + j for j in range(n)])
    for j in range(n):
        lines.append([n * i + j for i in range(n)])

    for difference in range(-(n - 1), n):
        lines.append(
            [n * i + j for i in range(n) for j in range(n) if i - j == difference]
        )
    for total in range(2 * n - 1):
        lines.append(
            [n * i + j for i in range(n) for j in range(n) if i + j == total]
        )

    width = n * n
    matrix = tuple(
        tuple(1 if k in line else 0 for k in range(width)) for line in lines
    )
    assert len(matrix) == 6 * n - 2
    assert all(len(row) == width for row in matrix)
    return matrix


def relu(value: int) -> int:
    return max(value, 0)


def integer_delta_relu(value: int, target: int) -> int:
    """Kronecker delta on integers, implemented using only ReLU gates."""
    absolute_difference = relu(value - target) + relu(target - value)
    return relu(1 - absolute_difference)


def fnn_output(board: Sequence[int], n: int, weights: Matrix) -> int:
    """Evaluate the fixed-weight FNN indicator on one flattened board."""
    if len(board) != n * n or any(x not in (0, 1) for x in board):
        raise ValueError("board must contain exactly N^2 binary entries")
    if len(weights) != 6 * n - 2:
        raise ValueError("weight matrix has the wrong height")

    line_sums = [sum(w * x for w, x in zip(row, board)) for row in weights]

    output = 1
    # Every row and every column must contain exactly one queen.
    for line_sum in line_sums[: 2 * n]:
        output *= integer_delta_relu(line_sum, 1)

    # Every diagonal may contain zero or one queen.
    for line_sum in line_sums[2 * n :]:
        output *= integer_delta_relu(line_sum, 0) + integer_delta_relu(line_sum, 1)

    return output


def classical_is_legal(board: Sequence[int], n: int) -> bool:
    """Independent test based on queen coordinates and pairwise attacks."""
    queens = [(k // n, k % n) for k, occupied in enumerate(board) if occupied]
    if len(queens) != n:
        return False

    for first in range(n):
        i1, j1 = queens[first]
        for second in range(first + 1, n):
            i2, j2 = queens[second]
            same_row = i1 == i2
            same_column = j1 == j2
            same_diagonal = abs(i1 - i2) == abs(j1 - j2)
            if same_row or same_column or same_diagonal:
                return False
    return True


def board_from_mask(mask: int, n: int) -> Board:
    return tuple((mask >> k) & 1 for k in range(n * n))


def format_board(board: Sequence[int], n: int) -> str:
    symbols = (".", "Q")
    return "\n".join(
        " ".join(symbols[board[n * i + j]] for j in range(n)) for i in range(n)
    )


@dataclass(frozen=True)
class VerificationResult:
    n: int
    total_boards: int
    classical_accepted: int
    fnn_accepted: int
    disagreements: int
    elapsed_seconds: float
    first_disagreement: Board | None
    first_classical_value: int | None
    first_fnn_value: int | None


def verify_n(n: int, *, stop_at_first: bool = False) -> VerificationResult:
    """Exhaustively compare both predicates over all 2^(N^2) boards."""
    weights = build_weight_matrix(n)
    total = 1 << (n * n)
    classical_accepted = 0
    fnn_accepted = 0
    disagreements = 0
    first_disagreement: Board | None = None
    first_classical_value: int | None = None
    first_fnn_value: int | None = None

    start = time.perf_counter()
    boards_checked = 0
    for mask in range(total):
        board = board_from_mask(mask, n)
        classical_value = int(classical_is_legal(board, n))
        network_value = fnn_output(board, n, weights)
        boards_checked += 1
        classical_accepted += classical_value
        fnn_accepted += network_value

        if classical_value != network_value:
            disagreements += 1
            if first_disagreement is None:
                first_disagreement = board
                first_classical_value = classical_value
                first_fnn_value = network_value
            if stop_at_first:
                break

    elapsed = time.perf_counter() - start
    return VerificationResult(
        n=n,
        total_boards=boards_checked,
        classical_accepted=classical_accepted,
        fnn_accepted=fnn_accepted,
        disagreements=disagreements,
        elapsed_seconds=elapsed,
        first_disagreement=first_disagreement,
        first_classical_value=first_classical_value,
        first_fnn_value=first_fnn_value,
    )


def write_csv(results: Iterable[VerificationResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "N",
                "total_boards_checked",
                "classical_accepted",
                "fnn_accepted",
                "disagreements",
                "elapsed_seconds",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.n,
                    result.total_boards,
                    result.classical_accepted,
                    result.fnn_accepted,
                    result.disagreements,
                    f"{result.elapsed_seconds:.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively verify the fixed-weight N-queens FNN."
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="board sizes to verify (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--stop-at-first",
        action="store_true",
        help="stop each board-size run as soon as a disagreement is found",
    )
    parser.add_argument("--csv", type=Path, help="optional destination for CSV results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(n < 1 for n in args.n):
        print("error: every N must be positive", file=sys.stderr)
        return 2

    results: list[VerificationResult] = []
    print("N  checked       classical  FNN        disagreements  seconds")
    print("-  ------------  ---------  ---------  -------------  -------")
    for n in args.n:
        result = verify_n(n, stop_at_first=args.stop_at_first)
        results.append(result)
        print(
            f"{n:<2} {result.total_boards:>12}  "
            f"{result.classical_accepted:>9}  {result.fnn_accepted:>9}  "
            f"{result.disagreements:>13}  {result.elapsed_seconds:>7.3f}"
        )
        if result.first_disagreement is not None:
            print("\nFirst disagreement:")
            print(format_board(result.first_disagreement, n))
            print(
                f"classical={result.first_classical_value}, "
                f"FNN={result.first_fnn_value}\n"
            )

    if args.csv is not None:
        write_csv(results, args.csv)

    return 1 if any(result.disagreements for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
