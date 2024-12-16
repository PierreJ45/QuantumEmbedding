"""
This script aims to build a minimal working example of a quantum embedding.
The target embedding is the one described in Figure 4 of https://arxiv.org/pdf/2001.03622
"""

import random

def createData(n):
    """
    create two lists : X and Y. X contains the feature (only one here) and Y the target value (1 or -1)
    The target is -1 if and only if the feature is between -1 and 1

    Parameters
    ----------
    n : int
        The number of samples to generate
    """
    X = [random.uniform(-2, 2) for _ in range(n)]
    Y = [1 if -1 < x < 1 else -1 for x in X]
    return X, Y

def quantumCircuit(x, Theta) :
    """
    Compute the result of the quantum circuit presented in figure 4
    x is the single feature and Theta the list of weights of the quantum circuit
    """
    pass

print(123)