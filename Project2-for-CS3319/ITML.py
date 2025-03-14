import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from processData import loadSplitedData
from tqdm import tqdm
from metric_learn import ITML_Supervised
from KNN import runKNN

def runITML(X_train, X_test, y_train, y_test, k_range, label=""):
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    scores = []
    itml = ITML_Supervised(n_constraints=200, verbose=1)
    itml.fit(X_train, y_train)
    X_train_ITML = itml.transform(X_train)
    X_test_ITML = itml.transform(X_test)
    scores = runKNN(X_train_ITML, X_test_ITML, y_train, y_test, k_range, 'euclidean', label)

if __name__ == '__main__':
    k_range = range(2, 20, 1)
    X_train, X_test, y_train, y_test = loadSplitedData()
    runITML(X_train, X_test, y_train, y_test, k_range, 'ITML_eculidean')
    print('All done!')