import numpy as np
import tensorflow as tf

def p_Wasserstein_distance(y_true, y_pred,p):
    if len(y_true) !=  len(y_pred):
        print(len(y_true), len(y_pred))
        raise ValueError("The two input arrays must have the same length")
    else:
        n = len(y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # Sort the two arrays
        y_true.sort()
        y_pred.sort()
        # Compute the AKE
        dist = 0
        for i in range(n):
            dist += (abs(y_true[i] - y_pred[i])) ** p
        dist = (dist / n) ** (1 / p)
        return dist

