## Based on Chapter 7 of EVT Natural Hazards
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Normalizer:
    '''
    Normalize Data using the exceedance based method of EVT Natural Hazards
    Input: Data , Quantile threshold
    Output: Normalized Data, Local Threshold Dictionary
    '''

    def __init__(self,data, threshold):
        self.data = data
        self.threshold = threshold

    def normalize(self):
        threshold = self.threshold
        new_tropical_data = []
        threshold_dict = {}
        for country in self.data.Country.unique():
            temp_df = self.data[self.data.Country == country]
            temp_losses = temp_df.Losses
            u_local = np.quantile(temp_losses, threshold)
            threshold_dict[country] = u_local
            new_losses = temp_losses[temp_losses > u_local]
            new_losses = new_losses / u_local
            new_tropical_data.extend(list(new_losses))
        print("Data were normalized using the exceedance based method with threshold: ", threshold)
        return np.array(new_tropical_data), threshold_dict