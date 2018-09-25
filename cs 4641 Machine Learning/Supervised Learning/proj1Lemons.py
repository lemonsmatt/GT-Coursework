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


import numpy as np
import pandas as pd
import pickle
from itertools import product
from joblib import Parallel, delayed
import multiprocessing
import time


def setup_parameters():
    tuned_parameters = []
    tuned_parameters.append([{'n_neighbors':[1,2,3,4,5,10,20,30]}, KNeighborsClassifier(), "knn"])
    tuned_parameters.append([{'criterion':["entropy"], 'max_depth':[1,3,5,10, 15, 20,30,40,50, 60], 'min_samples_leaf':[1,2,3,4,5,10,20]}, DecisionTreeClassifier(), "DecisionTree"])
    tuned_parameters.append([{'n_estimators':[5,25,50,75,90,100,120,135,150], 'learning_rate':[0.001, 0.01, 0.1, 1, 10], 'algorithm':["SAMME"]}, AdaBoostClassifier(DecisionTreeClassifier(max_depth=10)), "Boost-max_depth_10"])
    tuned_parameters.append([{'solver':['sgd'], 'learning_rate_init':[0.001, 0.01, 0.1], 'max_iter':[50,100,150,200,300, 500], 'momentum':[0.2, 0.5, 0.9]  , 'hidden_layer_sizes':[(13,13), (130,130,130),(130,130,130,130), (100,)]}, MLPClassifier(), "NN"])
    tuned_parameters.append([{'kernel': ['rbf'], 'max_iter': [10,50,100,500, 1000], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],'C': [0.01, 0.1, 0.2, 0.8, 0.7, 1, 3, 5, 10,15, 20]}, SVC(), "SVM_rbf"])
    tuned_parameters.append([{'kernel': ['linear'], 'max_iter': [10,50,100,500, 1000],'C': [0.01, 0.1, 0.2, 0.8, 0.7, 1, 3, 5, 10,15, 20]}, SVC(), "SVM_linear"])
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

    print(cross_score, train_score, test_score,len(y_train), train_time, test_time , param)
    row =[cross_score, train_score, test_score, len(y_train), train_time, test_time]
    for key in param.keys():
        row.append(param[key])
    return row




if __name__ == '__main__':
    # Loading the dataset
    datasets = ["winequality-white", "magic04"]
    for dataset in datasets:
        runtimes = [["algo", "max_test_acc", "train_time", "test_time"]]
        data = pd.read_csv(dataset+ "/" + dataset+".csv", delimiter=",")
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
        for x in range(1,10):
            X_train_s, _, y_train_s, _ = train_test_split(X_train_f, y_train_f, test_size=0.1*x, random_state=0)
            X_train_list.append(X_train_s)
            y_train_list.append(y_train_s)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
        # Set the parameters by cross-validation
        with open(dataset+"-train.pkl", "wb") as f:
            pickle.dump([X_train_list, y_train_list], f)
        with open(dataset+"-test.pkl", "wb") as f:
            pickle.dump([X_test, y_test], f)

        tuned_parameters = setup_parameters()
        num_cores = multiprocessing.cpu_count()
        for algo in tuned_parameters:
            output = [["cross_val_acc", "training_acc", "test_acc", "train_size", "train_time", "test_time"] + list(algo[0].keys())]
            for param in [dict(zip(algo[0], v)) for v in product(*(algo[0]).values())]:
                output = output + Parallel(n_jobs=num_cores)(delayed(run_classifer)(algo[1], param, X_train, y_train, X_test, y_test) for X_train, y_train in zip(X_train_list, y_train_list))
            d = pd.DataFrame(data=output[1:], columns=output[0])
            print(d)
            train_time = d.train_time.mean()
            test_time = d.test_time.mean()
            max_score = d.test_acc.max()
            runtimes.append([algo[2],max_score, train_time, test_time])
            d.to_csv(dataset+ "/"+dataset+"-" + algo[2] + ".csv")
        d = pd.DataFrame(data=runtimes[1:], columns=runtimes[0])
        print(d)
        d.to_csv(dataset+ "/"+dataset+"-" + "timer" + ".csv")