# Email Spam Classification | Logistic Regression & SGD

An applied machine learning project exploring email spam classification using logistic regression trained with stochastic gradient descent. Implements the optimization pipeline from scratch and investigates hyperparameter selection, cross-validation, and regularization techniques.

**Course:** COMP 551 – Applied Machine Learning  
**Institution:** McGill University  
**Project:** Mini Project 2 – Training and Evaluation Practices  
**Authors:** Ethan Gat, Anto Catanzaro, Marc Saliba

---

## Overview

This project investigates the implementation and evaluation of a binary classification model for identifying spam emails using the [UCI Spambase dataset](https://doi.org/10.24432/C53G6X).

The dataset contains 4,601 email instances, each represented by 57 numerical features describing word frequencies, character frequencies, and sequences of capital letters.

The primary objective was to develop a logistic regression classifier trained using stochastic gradient descent, then evaluate how optimization hyperparameters and regularization affect predictive performance and generalization.

Rather than relying entirely on existing machine learning implementations, the core logistic regression model, optimizer, and evaluation procedures were implemented from scratch using NumPy.

## Results

The final L2-regularized logistic regression model achieved **90.09% test accuracy** using a regularization strength of λ = 0.01.

| Metric | Result |
|---|---:|
| Test accuracy | 90.09% |
| Test cross-entropy | 0.3759 |
| Dataset size | 4,601 emails |
| Input features | 57 |
| Training split | 5% |
| Test split | 95% |
| Cross-validation | 5-fold |

The reported test results correspond to the fixed hyperparameter configuration used in the regularization experiment: batch size 16, learning rate 0.1, and 200 epochs.

## Methodology

### 1. Logistic Regression from Scratch

Implemented a binary logistic regression classifier using NumPy, including:

- Sigmoid activation for probability estimation.
- Binary cross-entropy loss.
- Analytical gradient computation.
- L2 regularization with an unregularized bias term.
- Probability prediction and classification accuracy.

### 2. Stochastic Gradient Descent

Developed a configurable SGD optimizer supporting:

- Mini-batch gradient descent.
- Configurable batch sizes and learning rates.
- Epoch-based training.
- Randomized sample ordering between epochs.
- Optional weight tracking for training convergence analysis.

The optimizer was implemented separately from the classifier to keep optimization logic independent of the model.

### 3. Hyperparameter Optimization

Performed a grid search over 180 configurations using five-fold cross-validation.

The search investigated:

- Batch size: 1, 16, 64.
- Learning rate: 0.0001, 0.001, 0.01, 0.1, 1.
- Epochs: 50, 150, 200.
- Weight initialization scale: 0, 0.01, 0.1, 1.

Cross-validation evaluated mean and standard deviation of cross-entropy and classification accuracy.

A one-standard-error selection procedure was used to identify a practical configuration, balancing validation performance with optimization efficiency.

The selected configuration was:

| Hyperparameter | Value |
|---|---:|
| Batch size | 64 |
| Learning rate | 0.1 |
| Epochs | 200 |
| Initialization scale | 0 |

### 4. L2 Regularization and Bias–Variance Analysis

Investigated the effect of regularization strength (λ) on training and validation performance.

A regularization sweep demonstrated the trade-off between overfitting and underfitting:

- Weak regularization allowed the model to fit training data closely, increasing the generalization gap.
- Strong regularization constrained model weights and increased bias.
- An intermediate λ = 0.01 produced the best validation cross-entropy in the experiments.

### 5. L1 Regularization and Feature Sparsity

Used scikit-learn's logistic regression implementation with L1 regularization to investigate coefficient sparsity.

Analyzed regularization paths to observe how feature coefficients changed as regularization strength decreased.

The experiments demonstrated the trade-off between model interpretability and predictive performance: stronger L1 regularization produced sparser models, while weaker regularization retained more features.

## Technical Stack

- **Python** – Implementation and experimentation
- **NumPy** – Model implementation, numerical computation, and optimization
- **Pandas** – Dataset processing and experiment results
- **Matplotlib** – Training curves, hyperparameter heatmaps, and regularization visualizations
- **scikit-learn** – L1-regularized logistic regression
- **ucimlrepo** – UCI dataset retrieval
- **Jupyter Notebook** – Experimental workflow and analysis

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── spambase_analysis.ipynb
├── src/
│   ├── logistic_regression.py
│   ├── optimizers.py
│   ├── preprocessing.py
│   ├── cross_validation.py
│   └── evaluation.py
├── results/
│   ├── figures/
│   └── metrics/
└── docs/
    └── report.pdf
```

## Getting Started

### Requirements

- Python 3.12+
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/spambase-classification.git
cd spambase-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

Open `notebooks/spambase_analysis.ipynb` and execute the cells sequentially.

The notebook retrieves the Spambase dataset directly from the UCI Machine Learning Repository.

## Limitations and Future Work

Several limitations provide opportunities for further investigation:

- **Limited training data:** Only 5% of the dataset was used for training, resulting in relatively noisy cross-validation estimates.
- **Hyperparameter selection:** Regularization strength was investigated separately rather than jointly optimized with all other hyperparameters.
- **Evaluation metrics:** Additional metrics, particularly precision and recall, would provide a more complete evaluation for spam filtering.
- **Model comparisons:** Additional classifiers could be evaluated against logistic regression using the same experimental protocol.
- **L1 convergence:** Solver convergence should be verified when evaluating the L1 regularization path.

Potential extensions include repeated cross-validation, joint randomized hyperparameter search, and comparisons with alternative classification algorithms.

## References

1. Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). [Spambase](https://doi.org/10.24432/C53G6X). UCI Machine Learning Repository.
2. Androutsopoulos, I., et al. (2000). [An Evaluation of Naive Bayesian Anti-Spam Filtering](https://arxiv.org/abs/cs/0006013).

---

*Developed collaboratively as part of COMP 551 at McGill University, October 2026.*