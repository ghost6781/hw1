""" 
polynomial regression on a single feature (X_n[:, 2]) 
Create a Python script polynomial regression 1d vis.py for the following.
It is difficult to visualize the results of high-dimensional regression. Instead, only use one
of the features (use the 3rd feature, i.e. X n[:,2] since Python index starts with 0) and
again perform polynomial regression. Produce plots of the training data points, learned
polynomial, and test data points.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
from utils import loadData, normalizeData


def poly_features_1d(x, deg):
    """Design matrix for a 1-D input: columns [x^0, x^1, ..., x^deg].

    x^0 is all ones, so it acts as the bias term. Shape: (N, deg + 1).
    """
    x = np.asarray(x).reshape(-1)
    return np.vstack([x ** k for k in range(deg + 1)]).T


if __name__ == "__main__":
    # ---- Load and normalize features and targets ----
    t, X = loadData()
    X_n = normalizeData(X)
    t = normalizeData(t).reshape(-1)

    # Use only the 3rd feature (index 2, since Python indexing starts at 0)
    x = X_n[:, 2]

    # ---- First 100 points train, the remainder test ----
    N_TRAIN = 100
    xtr, ttr = x[:N_TRAIN], t[:N_TRAIN]
    xte, tte = x[N_TRAIN:], t[N_TRAIN:]

    # Dense grid of x values to draw a smooth curve of the learned polynomial
    grid = np.linspace(x.min(), x.max(), 500)

    # Degrees to visualize: low (under-fit), middle, and high (over-fit)
    degrees = [1, 2, 3, 4, 6, 8, 10]

    for d in degrees:
        # Least-squares fit on the training points only
        w = np.linalg.lstsq(poly_features_1d(xtr, d), ttr, rcond=None)[0]

        # Report train and test MSE for this degree
        tr = np.mean((poly_features_1d(xtr, d) @ w - ttr) ** 2)
        te = np.mean((poly_features_1d(xte, d) @ w - tte) ** 2)
        print(f"degree {d:2d}  train MSE = {tr:.4f}  test MSE = {te:.4f}")

        # One figure per degree: data points plus the fitted polynomial
        plt.figure(figsize=(6, 4))
        plt.scatter(xtr, ttr, c="tab:blue", s=18, label="Training data")
        plt.scatter(xte, tte, c="tab:orange", s=18, marker="x", label="Test data")
        plt.plot(grid, poly_features_1d(grid, d) @ w, "k-", lw=2, label=f"Degree {d} fit")

        # Clip the y-range to the training targets (plus padding) so a wildly
        # oscillating high-degree curve doesn't squash the data points
        lo, hi = ttr.min(), ttr.max()
        pad = 0.5 * (hi - lo)
        plt.ylim(lo - pad, hi + pad)

        plt.xlabel("Feature 3 (normalized)")
        plt.ylabel("MPG (normalized)")
        plt.title(f"Degree {d}: train MSE {tr:.3f}, test MSE {te:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"polynomial_regression_1d_deg{d}.png", dpi=150)
        plt.close()  # free memory; we make several figures in a loop