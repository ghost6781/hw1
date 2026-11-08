"""
Create a Python script polynomial regression reg.py for the following.
Implement L2 -regularized regression. Again, use the first 100 points, 
and only use the 3rdfeature. Fit a degree 8 polynomial using λ = {0, 0.01, 0.1, 1, 10, 100, 1000}. 
Use 10-fold cross-validation to decide on the best value for λ. Produce a plot of average validation set error
versus regularizer value. Use a semilogx plot, putting regularizer value on a log scale2 .
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save the figure to a file
import matplotlib.pyplot as plt
from utils import loadData, normalizeData

DEGREE = 8                                  # polynomial degree fixed by the assignment
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100, 1000]  # candidate regularizer values


def poly_features_1d(x, deg):
    """Design matrix [x^0, x^1, ..., x^deg] for a 1-D input (x^0 = bias column)."""
    x = np.asarray(x).reshape(-1)
    return np.vstack([x ** k for k in range(deg + 1)]).T


def ridge_fit(Phi, t, lam):
    """Closed-form ridge solution w = (Phi^T Phi + lam*I)^-1 Phi^T t.

    The bias weight w_0 is NOT penalized (penalty entry set to 0): regularizing
    the intercept would pull predictions toward 0 for no good reason.
    lstsq is used instead of an explicit inverse for numerical stability
    (and it still works when lam = 0 and Phi^T Phi is nearly singular).
    """
    P = lam * np.eye(Phi.shape[1])  # lambda * I
    P[0, 0] = 0.0                   # leave the bias unpenalized
    A = Phi.T @ Phi + P
    return np.linalg.lstsq(A, Phi.T @ t, rcond=None)[0]


def mse(Phi, t, w):
    """Mean squared error of predictions Phi @ w against targets t."""
    return float(np.mean((Phi @ w - t) ** 2))


if __name__ == "__main__":
    # ---- Load, normalize, and keep the first 100 points / 3rd feature only ----
    t, X = loadData()
    X_n = normalizeData(X)
    t = normalizeData(t).reshape(-1)

    N_TRAIN = 100
    x, y = X_n[:N_TRAIN, 2], t[:N_TRAIN]
    Phi = poly_features_1d(x, DEGREE)  # (100, DEGREE + 1) design matrix

    # ---- 10-fold cross-validation setup ----
    # Split the 100 indices into 10 contiguous folds of 10. Contiguous is fine
    # because loadData already applied a fixed random permutation.
    K = 10
    folds = np.array_split(np.arange(N_TRAIN), K)

    avg_val = []  # average validation error for each lambda
    for lam in LAMBDAS:
        errs = []
        for k in range(K):
            val = folds[k]  # fold k is held out for validation
            # Train on the other 9 folds
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            w = ridge_fit(Phi[tr], y[tr], lam)
            errs.append(mse(Phi[val], y[val], w))  # error on the held-out fold
        avg_val.append(np.mean(errs))  # average over the 10 folds
        print(f"lambda = {lam:<7}  avg validation MSE = {avg_val[-1]:.4f}")

    # The best lambda is the one with the lowest average validation error
    best = LAMBDAS[int(np.argmin(avg_val))]
    print(f"Best lambda by 10-fold CV: {best}")

    # ---- Plot validation error vs lambda on a log x-axis ----
    # lambda = 0 cannot be placed on a log axis (log 0 is undefined), so plot
    # only lambda > 0 and show lambda = 0 as a horizontal reference line.
    pos = [(l, e) for l, e in zip(LAMBDAS, avg_val) if l > 0]
    plt.figure(figsize=(6, 4))
    plt.semilogx([p[0] for p in pos], [p[1] for p in pos], "o-", label="Avg. validation error")
    plt.axhline(avg_val[0], color="gray", ls="--", label=r"$\lambda=0$ (no regularization)")
    plt.xlabel(r"Regularizer $\lambda$ (log scale)")
    plt.ylabel("Average validation MSE")
    plt.title(f"10-fold CV, degree {DEGREE}, feature 3")
    plt.legend()
    plt.tight_layout()
    plt.savefig("polynomial_regression_reg.png", dpi=150)
    print("saved polynomial_regression_reg.png")