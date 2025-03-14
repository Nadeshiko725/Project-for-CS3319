import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm   
import concurrent.futures


def runSVM(X_train, X_test, y_train, y_test, kernel, C):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    # y_test = y_test.ravel()
    model = SVC(C=C, kernel=kernel)
    model.fit(X_train, y_train.ravel())
    acc = model.score(X_test, y_test.ravel())
    print('SVM model accuracy_score:', acc)
    return acc


def searchBestC(X_train, X_test, y_train, y_test, C_range, kernel, k):
    # 创建一个能存放C和对应score的列表
    cv_scores = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for C in C_range:
            future = executor.submit(calculate_score, X_train, y_train, X_test, y_test, C, kernel, k)
            futures.append(future)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            score, C = future.result()
            cv_scores.append((score, C))
            print('C:', C, 'score:', score)
    
    # bestC = C_range[cv_scores.index(max(cv_scores))]
    bestC = max(cv_scores, key=lambda x: x[0])[1]
    bestModel = SVC(C=bestC, kernel=kernel)
    bestModel.fit(X_train, y_train)
    y_test = y_test.ravel()
    bestScore = bestModel.score(X_test, y_test)
    print('Best C:', bestC)
    print('Best score:', bestScore)
    cv_scores.sort(key=lambda x: x[1])
    # plot
    plt.figure()
    plt.plot(C_range, [score for score, C in cv_scores], 'bo-', linewidth=2)
    plt.title('SVM with ' + kernel + ' kernel')
    plt.xlabel('C')
    plt.ylabel('cross_val_score')
    plt.savefig('SVM_' + kernel + '.jpg')
    # save cv_scores
    with open('res_linear_' + kernel + '.txt', 'w') as f:
            for i in range(len(C_range)):
                f.write(kernel + ": C = %f, acc = %f\n"%(C_range[i], cv_scores[i]))
            f.write(kernel + ": Best C = %f\n"%(bestC))
            f.write(kernel + ": Best Score = %f\n"%(bestScore))

def calculate_score(X_train, y_train, X_test, y_test, C, kernel, k):
    model = SVC(C=C, kernel=kernel)
    # process data to fit cross_val_score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = y_train.ravel()
    scores = cross_val_score(model, X_train, y_train, cv=k)
    return scores.mean(), C

if __name__ == '__main__':
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    # C_range = [0.001, 0.01, 0.1, 1, 10, 100]
    C_range = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    kernel = 'linear'
    k = 5
    searchBestC(X_train, X_test, y_train, y_test, C_range, kernel, k)
    # rumSVM(X_train, X_test, y_train, y_test, 0.1, 'linear')



