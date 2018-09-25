from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

from pandas.plotting import table
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import time


def setup_parameters():
    tuned_parameters = []
    tuned_parameters.append([{'solver':['sgd'], 'learning_rate_init':[0.001], 'max_iter':[50,100,150,200,300, 500, 700], 'momentum':[0.5]  , 'hidden_layer_sizes':[(100,)]}, MLPClassifier(), "NN"])
    return tuned_parameters

def run_classifer(clf, param, X_train, y_train, X_test, y_test):
    model = clf.set_params(**param)
    cross_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    start = time.time()
    train_score = model.fit(X_train, y_train).score(X_train, y_train)
    end = time.time()
    train_time = end - start
    start = time.time()
    test_score = model.fit(X_train, y_train).score(X_test,y_test)
    end = time.time()
    test_time = end - start

    print(1 - cross_score, 1 - train_score, 1 - test_score,len(y_train), train_time, test_time , param)
    row =[1 - cross_score, 1 - train_score, 1 - test_score, len(y_train), train_time, test_time]
    for key in param.keys():
        row.append(param[key])
    return row


def plot_NN_graph(df, var, title, xlabel, ylabel, file_name="graph.png"):
    fig, ax = plt.subplots()
    i = 0
    for key, grp in df.groupby(["learning_rate_init", "momentum"]):
        ax = grp.plot(ax=ax, x = var, y = "training_err", label = "Training")
        ax = grp.plot(ax=ax, x = var, y = "test_err", label = "Test")
        i = i + 1
    fig.suptitle(title, y=1.08)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()

    fig.set_figheight(4)
    fig.set_figwidth(10)

    fig.savefig(file_name, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Loading the dataset
    dataset = "magic04"
    clustering = ["8_kmeans", "em"]
    filenames = ["pca_0.5", "ica_1.0E-4", "rp_50","data", "IG"]
    for cluster in clustering:
        for filename in filenames:
            if filename is "ica_1.0E-4" and cluster is "8_kmeans":
                continue
            runtimes = [["algo", "max_test_err", "train_time", "test_time"]]
            data = pd.read_csv("out/" + dataset+ "/" + filename+ "_" + cluster + "_set.csv", delimiter=",")
            le = LabelEncoder()
            for col in data.columns.values:
                if data[col].dtypes == 'object':
                    le.fit(data[col].values)
                    data[col] = le.transform(data[col])



            data = data.as_matrix()
            print(len(data))
            print(data)
            # To apply an classifier on this data, we need to flatten the image, to
            # turn the data in a (samples, feature) matrix:
            n_samples = len(data)
            X = data[:,:-1]
            y = data[:,-1]

            # Split the dataset in two equal parts
            X_train_f, X_test, y_train_f, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            X_train_list = [X_train_f]
            y_train_list = [y_train_f]

            tuned_parameters = setup_parameters()
            num_cores = multiprocessing.cpu_count()
            for algo in tuned_parameters:
                output = [["cross_val_err", "training_err", "test_err", "train_size", "train_time", "test_time"] + list(algo[0].keys())]
                for param in [dict(zip(algo[0], v)) for v in product(*(algo[0]).values())]:
                    output = output + Parallel(n_jobs=num_cores)(delayed(run_classifer)(algo[1], param, X_train, y_train, X_test, y_test) for X_train, y_train in zip(X_train_list, y_train_list))
                data = pd.DataFrame(data=output[1:], columns=output[0])
                print(data)
                data.to_csv(dataset+ "/"+filename+"_NN.csv")
                data = data.reset_index(drop=True)
                max_index = data.cross_val_err.idxmax()
                print(data.ix[max_index])
                data.sort_values(by=["max_iter", "momentum", "learning_rate_init", "train_size"], inplace=True)
                data = data.drop('train_size', 1)
                data = data.drop('hidden_layer_sizes', 1)
                data = data.drop('solver', 1)
                plot_NN_graph(data, "max_iter", filename+" NN Learning Curve", "Epochs", "Error", file_name="out/" + dataset+"/"+filename+  "_" + cluster + "_set_learning_curve.png")



