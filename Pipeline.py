import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from Distribution_Shifting import Distribution_Shifting
from GAN_Models import DCGAN
from Normalization import Normalizer
from BatchSize_Optimization import batch_size_optimization
from Evaluation_metrics import p_Wasserstein_distance
from scipy.stats import skewnorm, genpareto
from Conditional_GAN import Conditional_DCGAN


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
        self.data, self.country_thresholds, self.countries_size_dict, self.orig_loss_data = normalizer.normalize()
        if model == 'DCGAN':
            self.model = model
        else:
            raise ValueError("Only DCGAN is supported at the moment")

    def batch_optimization(self, epochs=200, batch_size_list=[32, 64, 128, 256, 512], distance='1-Wasserstein'):
        self.best_batch_size = batch_size_optimization(self.data, batch_size_list, epochs, self.model, distance)
        if self.model == "DCGAN":
            self.model = DCGAN(self.data, self.best_batch_size, self.epochs, self.Gloss_function, self.Dloss_function)

    def Distribution_Shifting(self, c, k):
        self.c = c
        self.k = k
        self.shifted_data = Distribution_Shifting(self.data, c, k, self.model)
        return self.shifted_data

    def fit_GPD(self, extremeness_measure):
        self.extremeness_measure = extremeness_measure
        self.measures = self.extremeness_measure(self.shifted_data)
        self.threshold = self.measures.min()
        self.tail = self.measures[np.where(self.measures > self.threshold)[0]]
        self.genpareto_params = genpareto.fit(self.tail - self.threshold)

    def train_Conditional_GAN(self):
        self.conditional_gan = Conditional_DCGAN(data=self.shifted_data,
                                                 batch_size=self.best_batch_size,
                                                 epochs=self.epochs,
                                                 extremeness_measure=self.extremeness_measure,
                                                 gen_pareto_params=self.genpareto_params,
                                                 threshold=self.threshold,
                                                 c=self.c,
                                                 k=self.k,
                                                 Gloss_function=self.Gloss_function,
                                                 Dloss_function=self.Dloss_function)
        self.conditional_gan.train()

    def generate_samples(self, country, tau):
        try:
            number_of_samples = self.countries_size_dict[country]
        except:
            print("This country is not included in this climate zone. Pick another country")
            raise KeyError

        self.country = country
        self.gen_samples = self.conditional_gan.generate_samples(number_of_samples, tau)
        self.gen_samples = np.array(self.gen_samples).ravel() * self.country_thresholds[country]

        return self.gen_samples

    def plot_distribution_shifting_hist(self):
        plt.hist(self.shifted_data, bins=10, color='red')
        plt.title("Synthetic Data")
        plt.show()
        plt.hist(self.data, bins=10, color='red')
        plt.title("Real Data")
        plt.show()

    def plot_hist(self):
        plt.hist(self.gen_samples, bins=10, color='red')
        title = "Synthetic Data Generated for country " + self.country
        plt.title(title)
        plt.show()
        title2 = "Real Data for country " + self.country
        plt.hist(self.orig_loss_data[self.country], bins=10, color='red')
        plt.title(title2)
        plt.show()