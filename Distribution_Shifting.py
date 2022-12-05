import numpy as np
import pandas as pd
import math
from Evaluation_metrics import p_Wasserstein_distance

def Distribution_Shifting(data, c, k, GAN):
    X = np.array(data)
    X.sort()
    Xs = X
    n = len(X)
    print("Distribution Shifting Started")
    for i in range(1,k+1):
        print("Iteration: ", i)
        GAN.prepare_new_data(Xs)
        GAN.train()
        Xs = list(X[-math.floor(c**(i) * n):])
        num_of_samples_to_generate = math.ceil((1/c) * (n - math.floor(c**(i) * n)))
        generated_samples = np.array(GAN.generate_samples(num_of_samples_to_generate))
        generated_samples.sort()
        Xs.extend(generated_samples[-(n - math.floor(c**(i) * n)):])
        Xs = np.array(Xs)
        print(f"The 1-Wasserstein distance between initial dataset and iteration {i} is: ", p_Wasserstein_distance(X, Xs, 1))

    return Xs