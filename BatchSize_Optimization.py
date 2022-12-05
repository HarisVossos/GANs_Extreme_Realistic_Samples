from GAN_Models import DCGAN
import torch
import torch.nn as nn
import math
from Evaluation_metrics import p_Wasserstein_distance

def batch_size_optimization(data, batch_size_list, epochs, model, GLoss = nn.L1Loss(), DLoss = nn.L1Loss(), distance = '1-Wasserstein'):
    if distance == '1-Wasserstein':
        p = 1
    elif distance == '2-Wasserstein':
        p = 2
    else:
        raise ValueError("The distance must be either 1-Wasserstein or 2-Wasserstein")
    if model != 'DCGAN':
        raise ValueError("Only DCGAN is supported at the moment")

    smallest_dist = math.inf
    best_batch_size = None

    for batch_size in batch_size_list:
        # Train the GAN
        print("Training the GAN with batch size: ", batch_size)
        GAN = DCGAN(data, batch_size, epochs, GLoss, DLoss)
        GAN.train()
        # Generate the data
        synthetic_data = GAN.generate_samples(len(data))
        # Evaluation metric
        dist = p_Wasserstein_distance(data, synthetic_data, p)
        if dist < smallest_dist:
            smallest_dist = dist
            best_batch_size = batch_size
    print(f'The best batch size is {best_batch_size} with a {p}-Wasserstein distance of {round(smallest_dist,4)}')
    return best_batch_size