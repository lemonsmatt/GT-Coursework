from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.decomposition import FastICA

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import time
import arff



if __name__ == '__main__':
    # Loading the dataset
    datasets = ["winequality-white", "magic04"]
    for dataset in datasets:
        print(dataset)
        #k_means
        for data_types in  [["data"], ["IG"], ["ica_1.0E-2", "ica_1.0E-4", "ica_1.0E-7"], ["pca_0.5", "pca_0.75", "pca_0.95"], ["rp_25", "rp_50", "rp_75"]]:
            print(data_types)
            dfs = []
            for filename in data_types:
                dfs.append(pd.read_csv( dataset +"/" + filename +"_kmeans.csv", delimiter=",", header=None))
            fig, ax = plt.subplots()
            for df in dfs:
                df.plot(ax= ax, x=0, y=1)
            plt.title(dataset + " " + data_types[0].split("_")[0] + " Kmeans")
            plt.ylabel("SSE")
            plt.xlabel("k")
            if len(data_types) > 1:
                ax.legend([x.split("_")[1] for x in data_types])
            else:
                ax.legend().set_visible(False)

            fig.savefig(dataset +"/" + data_types[0].split("_")[0] +"_kmeans.png", bbox_inches='tight')

            plt.close()

        print("Kurtosis")
        df_kurt = pd.DataFrame()
        icas = ["ica_1.0E-2", "ica_1.0E-4", "ica_1.0E-7"]
        for ica in icas:
            data = arff.load(open( dataset +"/" + ica + ".arff", "rb"))
            data = np.array(data['data'])
            df = pd.DataFrame(data)
            df = df.kurtosis()
            df_kurt[ica] = df
        print(df_kurt)
        fig, ax = plt.subplots()
        df_kurt.plot(ax=ax, kind='bar')
        plt.ylabel("Kurtosis")
        plt.xlabel("attribute")
        plt.title(dataset + " ICA Kurtosis")
        fig.savefig(dataset +"/" + "ica_kurtosis.png", bbox_inches='tight')
        plt.close()


