# t-SNE from Scratch in NumPy

A from-scratch Python/NumPy implementation of t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear dimensionality reduction and data visualization, based on van der Maaten & Hinton (2008).

Includes an empirical evaluation on the Breast Cancer Wisconsin dataset across varying perplexity values.

---

## Overview

Rather than relying on `scikit-learn`, this project implements the mathematical mechanics of the t-SNE algorithm directly using vectorized NumPy operations.

### 1. Pairwise Affinities in High Dimensions (P)

- Vectorized computation of squared Euclidean distance matrices.

- Binary search to find the optimal Gaussian bandwidth (\(\sigma_i\)) per sample matching the target perplexity.

- Symmetrization of conditional probabilities.

### 2. Low-Dimensional Mapping (Q)

- Student-t distribution (Cauchy kernel) to address the crowding problem.

### 3. Optimization

- Analytical gradient calculation of KL divergence.

- Gradient descent with momentum updates.

- Early exaggeration.

- Real-time KL divergence monitoring.

---

## Experiments: Breast Cancer Wisconsin Dataset

The implementation was evaluated on the Breast Cancer Wisconsin dataset:

- 30 continuous features

- 569 instances

- Perplexities tested: **10, 20, 30, 45**

- 1000 iterations per experiment

---

## Repository Structure

- `tsne_scratch.ipynb` - Full t-SNE implementation and experiments

- `breast-cancer.csv` - Dataset used for evaluation

- `requirements.txt` - Project dependencies

---

## Quick Start

### 1. Clone the Repository

```bash

git clone https://github.com/ArisPagonopoulos/tsne-from-scratch-python.git

cd tsne-from-scratch-python

```

### 2. Install Dependencies

```bash

pip install -r requirements.txt

```

### 3. Run the Notebook

```bash

jupyter notebook tsne_scratch.ipynb

```
