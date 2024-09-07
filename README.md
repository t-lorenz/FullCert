# FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks

This repository contains the implementation of FullCert - a deterministic end-to-end certifier for training and
inference of neural networks, published at GCPR 2024.

**TLDR:**
FullCert is the first end-to-end certifier with sound, deterministic robustness guarantees against both training- and
inference-time attacks.

Full Paper: &nbsp; [[Homepage]](https://www.t-lorenz.com/publication/fullcert-deterministic-end-to-end-certification-for-training-and-inference-of-neural-networks) &nbsp; [[arXiv]](https://arxiv.org/abs/2406.11522)

The repository includes two parts:

1. Our new BoundFlow library, which allows end-to-end certification of neural networks
2. Our experiments on the MNIST and Two-Moons datasets

## Installation

You can install the dependencies either via pip or conda. We recommend using conda.

### conda

Setup the environment
```bash
conda env create -f environment.yml
```
Activate the new environment
```bash
conda activate fullcert
```

### pip

```bash
pip install -r requirements.txt
```

## Basic Usage

To train a model, use the `certified_training.py` script.
All scripts support the `--help` flag, which will print a description of all available options.

```bash
python certified_training.py --help
```

For example
```bash
python certified_training.py\
      --batch-size 100\
      --learning-rate 0.5\
      --epochs 20\
      --pretrain-epochs 0\
      --epsilon 0.01\
      --hidden-nodes 20\
      --dataset moons\
      --experiment example\
      --loss bce\
      --perturb-inference
```

## Reproducing Results from the Paper

You can reproduce the plots from Figure 3 using the following commands:

Figure 1a: Two Moons Dataset
```bash
python certified_training_figures.py\
  --batch-size 100\
  --learning-rate 0.05\
  --epochs 500\
  --pretrain-epochs 500\
  --pretrain-learning-rate 2.0\
  --pretrain-size 100\
  --epsilon 0.001\
  --hidden-nodes 20\
  --layers 3\
  --dataset moons\
  --perturbation all\
  --experiment figure1a\
  --seed 10123310\
  --loss bce\
  --perturb-inference
```

Figure 1b: MNIST Dataset
```bash
python certified_training_figures.py\
  --batch-size 100\
  --learning-rate 0.01\
  --epochs 20\
  --pretrain-epochs 300\
  --pretrain-learning-rate 1.0\
  --pretrain-size 100\
  --epsilon 0.0001\
  --hidden-nodes 10\
  --layers 3\
  --dataset mnist17\
  --perturbation all\
  --experiment figure1b\
  --seed 10123310\
  --loss bce\
  --perturb-inference
```

The scripts used to produce the tables from the appendix are located in `code/hyperparameter_experiments`.

## Citation

```bibtex
@inproceedings{lorenz2024fullcert,
    title          = {FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks}, 
    author         = {Tobias Lorenz and Marta Kwiatkowska and Mario Fritz},
    booktitle      = {German Conference on Pattern Recognition (GCPR)},
    year           = {2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
