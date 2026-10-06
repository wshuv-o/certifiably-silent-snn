"""Shared analysis helpers."""
import numpy as np


def crossing(s, r):
    """Activity level where log(T_SS/T_DS) changes sign from negative to positive
    (log-interpolated in s). inf: sparse never loses for s <= 1; 0: sparse never
    wins in the grid; nan: non-monotone (reported, not guessed)."""
    lr = np.log(r)
    for i in range(len(s) - 1):
        if lr[i] <= 0 < lr[i + 1]:
            t = -lr[i] / (lr[i + 1] - lr[i])
            return float(np.exp(np.log(s[i]) + t * (np.log(s[i + 1]) - np.log(s[i]))))
    if np.all(lr <= 0):
        return float("inf")
    if np.all(lr > 0):
        return 0.0
    return float("nan")
