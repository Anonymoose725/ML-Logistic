import numpy as np
import itertools

class StochasticGradientDescent:

    def __init__(self, batch_size=16, lr=0.1, num_epochs=200, logging=True):
        """
        Initializes StochasticGradientDescent optimizer.

        This class implements minibatch stochastic gradient descent.
        """

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = num_epochs
        self.logging = logging
        self.rng = np.random.default_rng()

        if self.logging:
          self.w_log = []

    def run(self, gradient_fn, x, y, w):
        """
        Runs stochastic gradient descent.

        Parameters:
        gradient: function that computes gradient given (x, y, w),
                      returns gradient with same shape as w
        x: NumPy array of features, shape (N, D)
        y: NumPy array of target, shape (N,)
        w: NumPy array of initial weights, shape (D,)

        Returns:
        w:  NumPy array of final weights, shape (D,)
        """
        grad = np.inf

        for epoch in range(self.epochs):
          if self.logging:
            self.w_log.append(w)

          indices = self.rng.permutation(x.shape[0]) # shuffles indices

          for batch in itertools.batched(indices, self.batch_size):
            batch = list(batch)
            grad = gradient_fn(x[batch], y[batch], w)
            w = w - self.lr * grad

        if self.logging:
            self.w_log.append(w)

        return w