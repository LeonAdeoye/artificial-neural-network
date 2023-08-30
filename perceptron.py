import numpy as np


class Perceptron(object):
    """Implements a perceptron classifier

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of mis-classifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features
        y : array-like, shape = [n_samples]
            Target values
        Returns
        -------
        self : object
        """

        self.w_ = np.zeros(1 + X.shape[1])  # Initialize weights to zero we add 1 for the zero-weight or bias
        self.errors_ = []  # Initialize errors to empty list

        for _ in range(self.n_iter):  # Loop through the number of iterations
            # Initialize errors to zero
            errors = 0
            # Loop through the training data. The zip() function, in each iteration, assigns the corresponding elements of X and y to the variables xi and target..
            # Say X = [[2, 5], [7, 3], [1, 8]] and y = [0, 1, 0]
            # First iteration: xi = [2, 5] and target = 0, Second iteration: xi = [7, 3] and target = 1, Third iteration: xi = [1, 8] and target = 0
            # Xi is a single row of the training sample with n features, and target is the corresponding class label.
            for xi, target in zip(X, y):
                # Update = learning rate * (target - prediction). Update is a scalar value that is added to the weights vector, and it is computed for each training sample so that the weights are updated incrementally.
                update = self.eta * (target - self.predict(xi))
                # Update the weights by selecting all elements of the array from row index 1 onwards (i.e., the feature weights) and adding the update value times the current training sample xi.
                # The update value is multiplied by xi because the input values are multiplied by the corresponding weights in the prediction, and we need to update the weights based on how much they are responsible for the error.
                # self._w[0] is th bias term, which is updated separately from the feature weights.
                # For example, if update = 0.1, and xi [0.2, 0.5] and wi = 0.5 then then it will be (0.2 * 0.1) + (0.5 * 0.1) + 0.5 = 0.57
                self.w_[1:] += update * xi
                # Update the bias term
                self.w_[0] += update
                # Increment the number of errors if the update value is not zero because then the prediction was wrong.
                errors += int(update != 0.0)
                # Append the number of errors for each iteration to the errors_ list.
                # we collect these values to study how well our perceptron performs during the training.
                self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # np.dot() calculates the vector dot product wTx
        # we do this because we want to compute the net input, which is the weighted sum of the inputs
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        # np.where() returns the class label 1 or -1 depending on the value of the net input
        # We do this because we want to use the predict method to predict the class label of a sample x
        return np.where(self.net_input(X) >= 0.0, 1, -1)




