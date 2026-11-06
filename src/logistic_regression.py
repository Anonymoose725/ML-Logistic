import numpy as np

class LogisticRegression:

    def __init__(self, lam=10**(-2), add_constant=True, init_scale=0.0):
        """
        Initializes LogisticRegression classifier.

        This class implements logistic regression.

        Parameters:
        lam: regularization strength, if 0, no regularization applied
        add_constant: if True, adds constant to weights
        init_scale: initial weight scale
        """

        self.lam = lam
        self.add_constant = add_constant
        self.init_scale = init_scale
        self.rng = np.random.default_rng()

    def fit(self, x, y, optimizer):
        """
        Fit the logistic regression model using the provided optimizer.

        Parameters:
        x: NumPy array of features, shape (N, D)
        y: NumPy array of target, must be binary, shape (N,)
        optimizer: optimizer with a run() function taking (gradient_fn, x, y, w)
                    and returning optimized weights
        """

        if self.add_constant:
          x = np.column_stack([np.ones(x.shape[0]), x])
          # keep constant term out front for convenience

        if self.init_scale > 0:
            self.w = self.rng.normal(0, self.init_scale, x.shape[1]) # initialize weights to init_scale, set to 0 by default
        else:
            self.w = np.zeros(x.shape[1]) # normal


        self.w = optimizer.run(self.get_gradient_fn(), x, y, self.w) # optimize weights

    def predict(self, x):
        """
        Predict class probabilities for input x.

        Parameters:
        x: NumPy array of features, shape (N,) or (N, D)

        Returns:
        y_hat: NumPy array of predicted probabilities, shape (N,)
        """

        if self.add_constant:
            x = np.column_stack([np.ones(x.shape[0]), x])

        y_hat = LogisticRegression.logistic(np.dot(x, self.w))

        return y_hat

    def get_gradient_fn(self):
      """
      Generates a gradient function with regularization lam.
      """

      def gradient_fn(x, y, w):
        y_hat = LogisticRegression.logistic(np.dot(x, w))

        grad = np.dot(x.T, y_hat - y) / x.shape[0]

        reg = self.lam * w
        if self.add_constant:
          reg[0] = 0 # excluding bias from regularization
        grad += reg # using lambda/2 convention

        return grad

      return gradient_fn

    def calculate_crossentropy(self, x, y, w):
      if self.add_constant:
        x = np.column_stack([np.ones(x.shape[0]), x])

      z = np.dot(x, w)
      #ce = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z))) (causes overflow)
      ce = np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))

      return ce


    def calculate_accuracy(self, x, y, w):
      if self.add_constant:
        x = np.column_stack([np.ones(x.shape[0]), x])

      y_hat = LogisticRegression.logistic(np.dot(x, w))
      y_pred = y_hat > 0.5

      acc = np.mean(y_pred == y)

      return acc

    @staticmethod
    def logistic(z):
      """
      Implements the logistic function.
      """
      return 1. / (1 + np.exp(-z))