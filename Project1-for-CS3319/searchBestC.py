from concurrent.futures import thread
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from processData import loadSplitedData

class searchBestC(thread.Thread):
    def __init__(self, X_train, X_test, y_train, y_test, C_range, kernel, k, tag):
        thread.Thread.__init__(self)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.C_range = C_range
        self.kernel = kernel
        self.k = k
        self.tag = tag
        self.best_score = 0
        self.best_C = 0

    def run(self):
        cv_socres = []
        for C in self.C_range:
            model = SVC(C=C, kernel=self.kernel)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.k)
            cv_socres.append(scores.mean())
        
        bestC = self.C_range[cv_socres.index(max(cv_socres))]
        bestModel = SVC(C=bestC, kernel=self.kernel)
        bestModel.fit(self.X_train, self.y_train)
        bestScore = bestModel.score(self.X_test, self.y_test)
    

