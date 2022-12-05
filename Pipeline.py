import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import os
from Distribution_Shifting import Distribution_Shifting
from GAN_Models import DCGAN
from Normalization import Normalizer
from BatchSize_Optimization import batch_size_optimization
from Evaluation_metrics import p_Wasserstein_distance
import matplotlib.pyplot as plt

class pipeline:
    def __init__(self, data, normalization_threshold, model, epochs, batch_size=128, Gloss_function=nn.L1Loss,
                 Dloss_function=nn.L1Loss):
        self.normalization_threshold = normalization_threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.best_batch_size = None
        self.Gloss_function = Gloss_function
        self.Dloss_function = Dloss_function
        normalizer = Normalizer(data, self.normalization_threshold)
        self.data, self.country_thresholds = normalizer.normalize()
        if model == 'DCGAN':
            self.model = model
        else:
            raise ValueError("Only DCGAN is supported at the moment")

    def batch_optimization(self, epochs=200, batch_size_list=[32, 64, 128, 256, 512], distance='1-Wasserstein'):
        self.best_batch_size = batch_size_optimization(self.data, batch_size_list, epochs, self.model, distance)
        self.model = DCGAN(self.data, self.best_batch_size, self.epochs, self.Gloss_function, self.Dloss_function)

    def Distribution_Shifting(self, c, k):
        self.shifted_data = Distribution_Shifting(self.data, c, k, self.model)
        plt.hist(self.shifted_data, bins='auto')
        plt.title("Synthetic Data")
        plt.show()
        plt.hist(self.data, bins='auto')
        plt.title("Real Data")
        plt.show()
        return self.shifted_data