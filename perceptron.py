import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Colormaps in matplotlib are used to map scalar data (typically in the form of intensities or values) to colors.
# They are often used in visualizations to represent different values with distinct colors.
# The ListedColormap class specifically allows you to create custom colormaps by specifying a list of colors that the colormap should use.
from matplotlib.colors import ListedColormap


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
        # We do this because we want to compute the net input, which is the weighted sum of the inputs
        # We use numpy to do the vector dot product because it is faster (vectorization) than the equivalent expression using a for loop and is easier to read and understand.
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return z

    def predict(self, X):
        """Return class label after unit step"""
        # np.where() returns the class label 1 or -1 depending on the value of the net input
        # We do this because we want to use the predict method to predict the class label of a sample x
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Extract the first 100 class labels that correspond to the 50 Iris-setosa and 50 Iris-versicolor flowers, respectively,
# and convert the class labels into the two integer class labels, 1 (versicolor) and -1 (setosa).
def prepare_data():
    # Read the iris dataset from the UCI Machine Learning Repository
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    #pd.set_option('display.max_rows', None)
    print(df)

    #        0    1    2    3               4
    # 0    5.1  3.5  1.4  0.2     Iris-setosa
    # 1    4.9  3.0  1.4  0.2     Iris-setosa
    # 2    4.7  3.2  1.3  0.2     Iris-setosa
    # 3    4.6  3.1  1.5  0.2     Iris-setosa
    # 4    5.0  3.6  1.4  0.2     Iris-setosa
    # ..   ...  ...  ...  ...             ...
    # 145  6.7  3.0  5.2  2.3  Iris-virginica
    # 146  6.3  2.5  5.0  1.9  Iris-virginica
    # 147  6.5  3.0  5.2  2.0  Iris-virginica
    # 148  6.2  3.4  5.4  2.3  Iris-virginica
    # 149  5.9  3.0  5.1  1.8  Iris-virginica

    # Below selecting the values from the first 100 rows of the fifth column (index 4) of the DataFrame df,
    # which likely corresponds to the class labels of the Iris dataset (e.g., 'Iris-setosa', 'Iris-versicolor', or 'Iris-virginica').
    # The values are converted into a NumPy array, which we assign to the vector y.
    y = df.iloc[0:100, 4].values
    print("Head of y:")
    print(y[:5])  # Print the first 5 rows

    # Print the tail of the matrix X (last few rows)
    print("\nTail of y:")
    print(y[-5:])    # Assign the first 100 class labels to the vector y where Iris-setosa is -1 and Iris-versicolor is 1

    y = np.where(y == 'Iris-setosa', -1, 1)
    print(y)
    # Assign the first 100 feature columns to the matrix X
    # We use the NumPy slicing method to select the first 100 rows and the columns at index 0 and 2 (i.e., sepal length and petal length)
    X = df.iloc[0:100, [0, 2]].values
    print("Head of X:")
    print(X[:5])  # Print the first 5 rows

    # Print the tail of the matrix X (last few rows)
    print("\nTail of X:")
    print(X[-5:])

    ppn = Perceptron(eta=0.1, n_iter=10)  # Create a perceptron object with a learning rate of 0.1 and 10 iterations
    ppn.fit(X, y)  # Fit the perceptron object to the training data

    # plot_scatter(X)
    # plot_errors(X, y, ppn)
    plot_decision_regions(X, y, ppn)


def plot_scatter(X):
    # The colon : before 50 indicates that all rows from the beginning up to (but not including) index 50 will be included. In other words, it selects the first 50 rows of the array.
    # The comma separates the row selection from the column selection.
    # 0: This specifies the column index. In this case, 0 refers to the first column of the array.
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')  # Plot the intersection of first 50 samples. The first feature is the y-axis and the second feature is the x-axis (e.g. 5.1 and 3.5 is one point). Give it a label setosa
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')  # Plot the last 50 samples with a label versicolor, colour blue and mark x.
    plt.xlabel('sepal length [cm]')  # Label the x-axis
    plt.ylabel('petal length [cm]')  # Label the y-axis
    plt.legend(loc='upper left')  # Add a legend in the upper left corner
    plt.show()


def plot_errors(X, y, classifier):
    # Generates a line plot with x-axis values ranging from 1 to the length of the ppn.errors_ list.
    # The y-axis values are taken from the ppn.errors_ list.
    # Each data point on the plot is marked with a circle ('o') marker.
    print(ppn.errors_)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')  # Plot the number of errors for each iteration
    plt.xlabel('Epochs')  # Label the x-axis
    plt.ylabel('Number of mis-classifications')  # Label the y-axis
    plt.show()

# TODO: Understand this function
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')  # Create a tuple of markers
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # Create a tuple of colours
    # cmap = ListedColormap(colors[:len(np.unique(y))]): This line creates a colormap using the ListedColormap class. It does the following:
    # np.unique(y): This uses the NumPy library to find the unique values in the array y.
    # Assuming that y is a one-dimensional array or list, np.unique(y) will return the unique elements in y.
    # len(np.unique(y)): This calculates the number of unique elements in y.
    # colors[:len(np.unique(y))]: This slices the colors tuple to include only the first n colors, where n is the number of unique elements in y.
    # ListedColormap(...): This creates a colormap using the specified colors.
    cmap = ListedColormap(colors[:len(np.unique(y))])  # Create a colour map

    # plot the decision surface
    # The first two lines of code create a meshgrid of two-dimensional coordinates that cover the feature space.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Get the minimum and maximum values of the first feature column
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Get the minimum and maximum values of the second feature column

    # np.arange(x1_min, x1_max, resolution): This creates a one-dimensional array of values ranging from x1_min to x1_max with a step size of resolution.
    # np.arange(x2_min, x2_max, resolution): This creates a one-dimensional array of values ranging from x2_min to x2_max with a step size of resolution.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),  np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # Predict the class labels for the corresponding two-dimensional points in the feature space.
    Z = Z.reshape(xx1.shape)  # Reshape the predicted class labels Z into a grid with the same dimensions as xx1 and xx2.
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)  # Draw a filled contour plot using the grid arrays and the class labels Z.
    plt.xlim(xx1.min(), xx1.max())  # Set the x-axis limits to the minimum and maximum values of the first feature column.
    plt.ylim(xx2.min(), xx2.max())  # Set the y-axis limits to the minimum and maximum values of the second feature column.

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):  # Enumerate over the unique class labels in y.
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),  marker=markers[idx], label=cl)  # Plot the class samples.

    plt.xlabel('sepal length [cm]')  # Label the x-axis
    plt.ylabel('petal length [cm]')  # Label the y-axis
    plt.legend(loc='upper left')  # Add a legend in the upper left corner
    plt.show()




