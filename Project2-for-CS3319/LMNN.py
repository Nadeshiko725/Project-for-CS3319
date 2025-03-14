from re import X
from networkx import neighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from processData import loadSplitedData
from tqdm import tqdm
from metric_learn import LMNN
from KNN import runKNN

def runLDA(X_train, X_test, y_train, y_test, n_comp):
    print('Start LDA with n_comp =', n_comp)
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(X_train, y_train)
    X_train_LDA = lda.transform(X_train)
    X_test_LDA = lda.transform(X_test)
    return X_train_LDA, X_test_LDA

def runLMNN(X_train, X_test, y_train, y_test, k_range, neighbor, label=""):
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    scores = []
    lmnn = LMNN(n_neighbors=neighbor, learn_rate=1e-6, verbose=1)
    lmnn.fit(X_train, y_train)
    print("LMNN fit finished")
    X_train_LMNN = lmnn.transform(X_train)
    X_test_LMNN = lmnn.transform(X_test)
    scores = runKNN(X_train_LMNN, X_test_LMNN, y_train, y_test, k_range, 'euclidean', label)

def main():
    k_range = range(2, 20, 1)
    neighbors_range = range(2, 11, 2)
    X_train, X_test, y_train, y_test = loadSplitedData()
    scores = np.zeros((len(neighbors_range), len(k_range)))
    X_train_LDA, X_test_LDA = runLDA(X_train, X_test, y_train, y_test, 49)
    for n in tqdm(neighbors_range, desc='Progress for LMNN', leave=True, colour='YELLOW'):
        scores[n-3] = runLMNN(X_train_LDA, X_test_LDA, y_train, y_test, k_range, n, 'LMNN_eculidean_neigh=' + str(n))
        print('LMNN using ' + str(n) + 'done!')

    plt.figure()
    for i in range(len(neighbors_range)):
        plt.plot(k_range, scores[i], label='n_neighbors=' + str(neighbors_range[i]))
    plt.xlabel('Value of K for KNN using euclidean distance')
    plt.ylabel('Testing Accuracy')
    plt.title('Accuracy on Test Data for Different K Values using euclidean distance')
    plt.legend()
    plt.savefig('knn_accuracy_LMNN.png')


    print('All done!')

if __name__ == '__main__':
    main()