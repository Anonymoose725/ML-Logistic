import numpy as np
from src.preprocessing import Standardizer
from src.logistic_regression import LogisticRegression
from src.optimizers import StochasticGradientDescent

# k-folding
def kfold(n, k, rng):
  """
  Lazily generates indices that split a numpy array into K folds, K-1 for training and 1 for validation.
  Computes pairs of indices rather than actual

  Parameters:
  n: number of samples
  k: number of desired folds
  rng: numpy random generator

  Returns:
  (train, validate) index pairs for each of K possible fold arrangements
  """
  idx = rng.permutation(n) # permute the order before folding
  folds = np.array_split(idx, k)

  for i in range(k): # exclusive for k
    validate = folds[i]

    train_parts = []
    for j in range(k):
      if j != i:
        train_parts.append(folds[j])
    train = np.concatenate(train_parts)

    yield train, validate


# cross validation properly ~ x[tr_idx] standardization generates mean and stddev, used to standardize s[val_idx]
def cross_validate(x, y, k_folds, learn_rate=0.1, lam=0.1, batch_size=16, epochs=200, init_scale=0.0):
  """
  Perform k-fold cross-validation with stochastic gradient descent

  Parameters:
  x:          np.ndarray, shape (N, D), instances, **NOT YET STANDARDIZED**
  y:          np.ndarray, shape (N,) binary training labels
  k_folds:    list of tuples (train, validate) indices that split the train set from validate
  init_scale: float, initial weight scale

  Hyperparameters to tune:
  learn_rate: learning rate for each step in SGD e.x. {1, 0.1, 0.001, 0.0001, 0.00001}
  lam:        lambda, regularization penalty e.x. {0.01, 0.1, 0.5}
  batch_size: size of each minibatch in SGD e.x. {1, 16, 64}
  epochs:     number of full passes over the training folds e.x. {50, 100, 150, 200}

  Returns: {ce_mean, ce_stddev, acc_mean, acc_stddev}, cross-entropy and accuracy metrics across k-fold arrangements
  """
  cross_entropies = []
  accuracies = []

  for tr_idx, val_idx in k_folds:
    # fit standardizer for every fold arrangement without validation set in consideration
    s = Standardizer()
    s.fit(x[tr_idx])

    x_train = s.standardize(x[tr_idx])
    x_validate = s.standardize(x[val_idx]) # using same parameters as train set
    y_train = y[tr_idx]
    y_validate = y[val_idx]

    model = LogisticRegression(lam, init_scale=init_scale)
    w = StochasticGradientDescent(batch_size, learn_rate, epochs, logging=False)
    model.fit(x_train, y_train, w)

    # metrics
    ce = model.calculate_crossentropy(x_validate, y_validate, model.w)
    acc = model.calculate_accuracy(x_validate, y_validate, model.w)

    cross_entropies.append(ce)
    accuracies.append(acc)

  # mean and standard deviations
  ce_mean = np.mean(cross_entropies)
  ce_stddev = np.std(cross_entropies)
  acc_mean = np.mean(accuracies)
  acc_stddev = np.std(accuracies)

  return {'ce_mean': ce_mean, 'ce_stddev': ce_stddev, 'acc_mean': acc_mean, 'acc_stddev': acc_stddev}


def get_best_hyperparameters(results, init_scale=0.0, n_folds=5, logging=False):
  """
  Retrieve the best configuration found through cross validation

  Rules to pick the best:
  1. keep only the runs with given init_scale
  2. average ce_mean over epochs, giving a single estimate per (B, lr) cell
  3. score a cell by averaging it between its neighbours learning rates
  4. any cell within one SE is a tie
  5. finally we pick the largest B, then smallest lr (the fastest while also most convergent solution)
  6. for the chosen cell, use the epoch count with the lowest CE mean, done!
  See https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html
  for the "one standard error" approach

  Parameters:
  results:        DataFrame of results post-CV
                  (columns B, lr, epochs, init_scale, ce_mean, ce_stddev, acc_mean, acc_stddev)
  init_scale: initialization scale to select within
  n_folds:    K used in cross validation, for the standard error

  Returns: {B, lr, epochs, init_scale}, best configuration
  """
  df = results[results.init_scale == init_scale] # satisfy (1)

  # (2) one row per B, one column per lr (ascending)
  cells = df.groupby(["B", "lr"]).ce_mean.mean().unstack("lr").sort_index(axis=1)
  stddevs = df.groupby(["B", "lr"]).ce_stddev.mean() # fold-to-fold stddev per cell

  # (3) mean of each cell and its lr neighbours, edge columns are excluded
  ties_resolved = cells.T.rolling(3, center=True, min_periods=3).mean().T # rolling = sliding window of 3, i.e. left + mid + right
  scores = ties_resolved.stack() # drops those excluded, indexed by (B, lr)

  # (4) one standard error approach
  B_best, lr_best = scores.idxmin()
  std_err = stddevs.loc[(B_best, lr_best)] / np.sqrt(n_folds)
  tied = scores[scores <= scores.min() + std_err]

  # (5) largest B and smallest lr, of those that tied in 1SE
  tied_df = tied.reset_index(name="score").sort_values(["B", "lr"], ascending=[False, True]) # B up, lr down
  B, lr = tied_df.iloc[0].B, tied_df.iloc[0].lr

  # (6) epoch count with the lowest ce_mean for the chosen (B, lr) panel
  panel = df[(df.B == B) & (df.lr == lr)]
  epochs = panel.loc[panel.ce_mean.idxmin(), "epochs"]

  if logging:
    chosen = panel[panel.epochs == epochs].iloc[0]
    print(f"{len(tied)} (B, lr) cells within one standard error of the best score")
    print(f"chosen: ce_mean={chosen.ce_mean:.3f}, ce_stddev={chosen.ce_stddev:.3f}, "
          f"acc_mean={chosen.acc_mean:.3f}")
  return {
      "B": int(B), "lr": float(lr), "epochs": int(epochs), "init_scale": float(init_scale)
  }


# cv helper that returns training metrics too
def cross_validate_with_train(x, y, k_folds, learn_rate=0.1, lam=0.1, batch_size=16, epochs=200, init_scale=0.0):
  """
  Perform k-fold cross-validation with stochastic gradient descent

  Parameters:
  x:          np.ndarray, shape (N, D), instances, **NOT YET STANDARDIZED**
  y:          np.ndarray, shape (N,) binary training labels
  k_folds:    list of tuples (train, validate) indices that split the train set
              from validate

  Hyperparameters to tune:
  learn_rate: learning rate for each step in SGD e.x. {1, 0.1, 0.001, 0.0001, 0.00001}
  lam:        lambda, regularization penalty e.x. {0.01, 0.1, 0.5}
  batch_size: size of each minibatch in SGD e.x. {1, 16, 64}
  epochs:     number of full passes over the training folds e.x. {50, 100, 150, 200}

  Returns: {val_ce_mean, val_ce_std, val_acc_mean, val_acc_std, tr_ce_mean,
            tr_ce_std, tr_acc_mean, tr_acc_std},
            cross-entropy and accuracy metrics across k-fold arrangements
  """
  # One list per metric; each list 1 val per fold
  metrics = {"val_ce": [], "val_acc": [], "tr_ce": [], "tr_acc": []}
  for tr_idx, val_idx in k_folds:

    # standardize using only this fold's training data (to avoid leakage)
    s = Standardizer()
    s.fit(x[tr_idx])
    s.std[s.std == 0] = 1.0 # avoid divide by zero
    xtr, xva = s.standardize(x[tr_idx]), s.standardize(x[val_idx])
    ytr, yva = y[tr_idx], y[val_idx]

    # train fresh model for this fold
    model = LogisticRegression(lam, init_scale=init_scale)
    model.fit(xtr, ytr, StochasticGradientDescent(batch_size, learn_rate,
                                                  epochs, logging=False))
    # logging=False doesn't store weights after every epoch to save memory


    # eval the trained weights on both parts and store 1 val per fold
    metrics["val_ce"].append(model.calculate_crossentropy(xva, yva, model.w))
    metrics["val_acc"].append(model.calculate_accuracy(xva, yva, model.w))
    metrics["tr_ce"].append(model.calculate_crossentropy(xtr, ytr, model.w))
    metrics["tr_acc"].append(model.calculate_accuracy(xtr, ytr, model.w))

  # summarize, then output
  output = {}
  for k, v in metrics.items():
    output[k + "_mean"] = np.mean(v)
    output[k + "_std"] = np.std(v)

  return output



