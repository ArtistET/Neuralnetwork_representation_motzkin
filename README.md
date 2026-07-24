# Verification of the Motzkin NQS constructions

This repository contains the small-system verification code accompanying the
Motzkin neural-quantum-state constructions. It implements the revised RNN,
FNN, CNN, and Transformer formulas and compares their unnormalized support
with the Motzkin-path definition by exhaustive enumeration.
## Requirements

- Python 3.10 or later
- NumPy

Install the dependency with

```bash
python -m pip install -r requirements.txt
```

## Reproduce the verification

From this directory, run

```bash
python run.py --sizes 4 6 --colors 1 2 3
```

The command exhaustively checks all four architectures for colorless
(`s=1`) and colorful (`s=2,3`) Motzkin states at `N=4,6`. It writes the result
directly to the terminal and exits with a nonzero status if any case fails.

The RNN, FNN, and CNN implementations use exact integer arithmetic. The
colorful Transformer retains a finite softmax, with
`omega=100` and `beta=100*N*omega`; its result is therefore reported as
agreement to machine precision rather than as a hard-attention identity.

## Files

- `reference.py`: independent combinatorial Motzkin-state definition
- `rnn.py`, `fnn.py`, `cnn.py`, `transformer.py`: literal implementations of
  the revised manuscript constructions
- `run.py`: exhaustive comparison and terminal summary

## Verification result

The exhaustive run above checks 24 cases. All cases pass: each architecture
agrees with the exact Motzkin support for `s=1,2,3` and `N=4,6`. The largest
basis checked is the colorful `s=3`, `N=6` space of dimension `7^6=117649`.
