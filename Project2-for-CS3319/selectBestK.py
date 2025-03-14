from pdb import run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from processData import loadSplitedData
from tqdm import tqdm
from KNN import runKNN

def runSelectBestK(X_train, X_test, y_train, y_test, k_range):
    cv_scores = []
    for k in tqdm(k_range, desc='Progress for selecting best k', leave=True, colour='YELLOW'):
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=5)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # Find the best k
    best_k = k_range[np.argmax(cv_scores)]
    best_acc = cv_scores[np.argmax(cv_scores)]
    print('Best k: ', best_k)
    print('Best accuracy: ', best_acc)

    # Save the scores
    with open('res_KNN_cv_scores.txt', 'w') as f:
        for i in range(len(k_range)):
            f.write('k=' + str(k_range[i]) + ' cv_accuracy=' + str(cv_scores[i]) + '\n')
        f.write('Best k: ' + str(best_k) + '\n')
        f.write('Best accuracy: ' + str(best_acc) + '\n')

    # Plot the results
    plt.plot(k_range, cv_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-validated Accuracy')
    plt.title('Cross-validated Accuracy on Training Data for Different K Values')
    plt.savefig('knn_cv_accuracy.png')
    plt.show()

def runBestK(X_train, X_test, y_train, y_test, best_k):
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=8)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print('Best k: ', best_k)
    print('Accuracy on test data: ', score)

    return score

def main():
    k_range = range(2, 50, 1)
    X_train, X_test, y_train, y_test = loadSplitedData()
    runSelectBestK(X_train, X_test, y_train, y_test, k_range)
    print('All done!')

if __name__ == '__main__':
    # main()
    X_train, X_test, y_train, y_test = loadSplitedData()
    runBestK(X_train, X_test, y_train, y_test, 7)