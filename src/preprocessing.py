import numpy as np

class Standardizer:
  def fit(self, data):
    """
    Calculates and stores the mean and standard deviation of the provided data.

    Parameters:
    data: NumPy array of data to fit
    """
    self.mean = np.mean(data, axis=0, keepdims=True)
    self.std = np.std(data, axis=0, keepdims=True)
    self.std[self.std == 0] = 1.0 # guard from zero to avoid issues down the line


  def standardize(self, data):
    """
    Standardizes the provided data using the stored mean and standard deviation.

    Parameters:
    data: NumPy array of data to standardize

    Returns:
    s_data: NumPy array of standardized data
    """
    s_data = (data - self.mean) / self.std

    return s_data
