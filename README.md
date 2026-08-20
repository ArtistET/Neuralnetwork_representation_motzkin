# Exact Neural-Network Representations of the Motzkin States

This repository provides exact neural-network constructions for the Motzkin chain, including RNN, FNN, CNN, and Transformer representations, as described in the paper *Exact Neural-Network Representations of the Motzkin States*. It also contains small-system verification code that compares these constructions with the exact Motzkin states. If you use this code or the constructions, please cite the paper:
```bibtex
@misc{zha2026exactneuralnetworkrepresentationsmotzkin,
      title={Exact Neural-Network Representations of the Motzkin States}, 
      author={Runde Zha and Yuntian Gu and Chaohui Fan and Jia-lin Chen and Hai-Jun Liao and Tao Xiang},
      year={2026},
      eprint={2607.22522},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el},
      url={https://arxiv.org/abs/2607.22522}, 
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

## N-queens NQS representation
Except for Motzkin chains, we showed a simple example for N-queens of FNN architecture in fnn_n_queens_verify.py, one can run

```bash
python fnn_n_queens_verify.py --n 1 2 3 4
```

The command checks the output of the FNN architecture at `N=1,2,3,4`.

