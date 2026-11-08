
"""
Polynomial Basis Function Regression

Assignment:
- Use the first 100 data points as training data.
- Use all remaining data points as testing data.
- Fit polynomial regression models for degrees 1 through 10.
- Do not use regularization.
- Calculate the training and testing mean squared errors (MSE).
- Plot both errors against polynomial degree.
"""

import numpy as np

# Use the non-interactive Agg backend so Matplotlib can save the graph
# to a file without opening a graphical window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the provided functions for loading and normalizing the dataset.
from utils import loadData, normalizeData


def poly_features(X, deg):
    """
    Generate polynomial features up to the specified degree.

    X: Input features.
    deg: Maximum polynomial degree.

    Returns a feature matrix containing:
    - A column of ones for the bias/intercept.
    - The original features raised to powers 1, 2, ..., deg.

    Note: This implementation does not create cross-terms between features.
    """

    # Ensure that X has the expected two-dimensional shape.
    # If X is one-dimensional, reshape it into a column vector.
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X

    # Add a column of ones to represent the bias/intercept term.
    cols = [np.ones((X.shape[0], 1))]

    # Generate polynomial terms for each degree from 1 to deg.
    # For example, if deg = 3, the terms are X, X^2, and X^3.
    for k in range(1, deg + 1):
        cols.append(X ** k)

    # Combine all generated feature columns into one matrix.
    return np.hstack(cols)


def fit(Phi, t):
    """
    Fit the polynomial regression model using least squares.

    Phi: Polynomial feature/design matrix.
    t: Target values.

    Returns the learned weight vector w.

    No regularization is applied.
    """

    # Solve Phi @ w ≈ t using NumPy's least-squares solver.
    # lstsq uses a numerically stable method and is helpful when
    # polynomial features become highly correlated at large degrees.
    return np.linalg.lstsq(Phi, t, rcond=None)[0]


def mse(Phi, t, w):
    """
    Calculate the mean squared error (MSE).

    Phi: Feature matrix.
    t: Actual target values.
    w: Learned model weights.

    MSE is the average of the squared differences between
    predicted values and actual target values.
    """

    # Predict the target values using the learned weights.
    predictions = Phi @ w

    # Calculate and return the average squared prediction error.
    return float(np.mean((predictions - t) ** 2))


if __name__ == "__main__":

    # Load the dataset.
    # t contains the target values (fuel efficiency).
    # X contains the input features describing the cars.
    t, X = loadData()

    # Normalize the input features to put them on comparable scales.
    X_n = normalizeData(X)

    # Normalize the target values and flatten them into a 1D array.
    t = normalizeData(t).reshape(-1)

    # Specify that the first 100 data points will be used for training.
    N_TRAIN = 100

    # Split the dataset without shuffling:
    # The first 100 observations are used to train the models.
    Xtr, ttr = X_n[:N_TRAIN], t[:N_TRAIN]

    # All observations after the first 100 are used for testing.
    Xte, tte = X_n[N_TRAIN:], t[N_TRAIN:]

    # Test polynomial degrees from 1 to 10, inclusive.
    degrees = list(range(1, 11))

    # Create empty lists to store training and testing MSE values.
    train_err, test_err = [], []

    # Train and evaluate one model for each polynomial degree.
    for d in degrees:

        # Generate polynomial features for the training data
        # and learn the model weights using least squares.
        w = fit(poly_features(Xtr, d), ttr)

        # Calculate the training MSE using the training data.
        train_err.append(mse(poly_features(Xtr, d), ttr, w))

        # Calculate the testing MSE using unseen testing data.
        # The same learned weights are used; the model is not retrained.
        test_err.append(mse(poly_features(Xte, d), tte, w))

        # Print the degree and its corresponding training and testing errors.
        print(
            f"degree {d:2d}  "
            f"train MSE = {train_err[-1]:.4f}  "
            f"test MSE = {test_err[-1]:.4f}"
        )

    # Plot the training and testing errors against polynomial degree.
    plt.figure(figsize=(6, 4))

    # Plot training MSE using circle markers and a solid line.
    plt.plot(degrees, train_err, "o-", label="Training error")

    # Plot testing MSE using square markers and a solid line.
    plt.plot(degrees, test_err, "s-", label="Test error")

    # Use a logarithmic scale for the vertical axis so that
    # differences between small and large errors are easier to see.
    plt.yscale("log")

    # Label the horizontal and vertical axes.
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean squared error (log scale)")

    # Add a descriptive title and a legend identifying each curve.
    plt.title("Error vs. polynomial degree (all 7 features)")
    plt.legend()

    # Adjust spacing so labels and the title fit within the figure.
    plt.tight_layout()

    # Save the graph as a PNG image in the current working directory.
    plt.savefig("polynomial_regression_a.png", dpi=150)

    # Confirm that the graph has been saved.
    print("saved polynomial_regression_a.png")
