# Exact Neural-Network Representations of the Motzkin States

This repository provides exact neural-network constructions for the Motzkin chain, including RNN, FNN, CNN, and Transformer representations, as described in the paper *Exact Neural-Network Representations of the Motzkin States*. It also contains small-system verification code that compares these constructions with the exact Motzkin states. If you use this code or the constructions, please cite the paper:
```bibtex
@misc{zha2026ExactMotzkinNQS,
  title     = {Exact Neural-Network Representations of the Motzkin States},
  author    = {Runde Zha, Yuntian Gu, Chaohui Fan, Jia-Lin Chen, Hai-Jun Liao, and Tao Xiang},
  year      = {2026}
}
```

## Requirements

- Python 3.10 or later
- NumPy

Install the dependency with

```bash
python -m pip install -r requirements.txt
```

## Files

- `reference.py`: independent combinatorial Motzkin-state definition
- `rnn.py`, `fnn.py`, `cnn.py`, `transformer.py`: literal implementations of
  the revised manuscript constructions
- `run.py`: exhaustive comparison and terminal summary

## Verification

```bash
python run.py --sizes 4 6 --colors 1 2 3
```
The command checks all four architectures for colorless (`s=1`) and colorful (`s=2,3`) Motzkin states at `N=4,6`.

