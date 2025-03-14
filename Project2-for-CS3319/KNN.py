import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import DistanceMetric
from sklearn.preprocessing import StandardScaler
from processData import loadSplitedData
from tqdm import tqdm

def runKNN(X_train, X_test, y_train, y_test, k_range, metric, label=""):
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    scores = []
    for k in tqdm(k_range, desc='Progress for ' + label, leave=True, colour='RED'):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=5)
        knn.fit(X_train, y_train)
        print("KNN fit finished")
        score = knn.score(X_test, y_test)
        print("k=", k, "score=", score)
        scores.append(score)
    
    # Plot the results
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN using ' + metric + ' distance')
    plt.ylabel('Testing Accuracy')
    plt.title('Accuracy on Test Data for Different K Values using ' + metric + ' distance')
    plt.savefig('knn_accuracy_' + label + '.png')
    
    # Find the best k
    best_k = k_range[np.argmax(scores)]
    best_acc = scores[np.argmax(scores)]
    print('Best k: ', best_k)
    print('Best accuracy: ', best_acc)
    
    # Save the model
    with open('res_KNN_model_' + label + '.txt', 'w') as f:
        for i in range(len(k_range)):
            f.write(label + ' k=' + str(k_range[i]) + ' accuracy=' + str(scores[i]) + '\n')
        f.write('Best k: ' + str(best_k) + '\n')
        f.write('Best accuracy: ' + str(best_acc) + '\n')
    return scores

if __name__ == '__main__':
    metric_range = ['euclidean', 'manhattan', 'minkowski']
    k_range = range(2, 20, 1)
    X_train, X_test, y_train, y_test = loadSplitedData()
    for metric in metric_range:
        runKNN(X_train, X_test, y_train, y_test, k_range, metric, metric)
        print('KNN using ' + metric + 'done!')
    print('All done!')