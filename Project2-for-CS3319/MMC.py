from msilib.schema import Component
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from processData import loadSplitedData
from tqdm import tqdm
from metric_learn import MMC_Supervised
from KNN import runKNN

def runMMC(X_train, X_test, y_train, y_test, k_range, label=""):
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    scores = []
    mmc = MMC_Supervised(n_constraints=200, verbose=1, diagonal=True)
    mmc.fit(X_train, y_train)
    X_train_MMC = mmc.transform(X_train)
    X_test_MMC = mmc.transform(X_test)
    scores = runKNN(X_train_MMC, X_test_MMC, y_train, y_test, k_range, 'euclidean', label)

if __name__ == '__main__':
    k_range = range(2, 20, 1)
    X_train, X_test, y_train, y_test = loadSplitedData()
    runMMC(X_train, X_test, y_train, y_test, k_range, 'MMC_eculidean')