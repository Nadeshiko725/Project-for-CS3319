import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot():
    # 读取当前目录下名为"res_KNN_model_LMNN_eculidean_neigh=2.txt"到neigh=8的文件，将其数据读出画在一张图上
    neigh_range = range(2, 9, 2)
    k_range = range(2, 20, 1)
    scores = np.zeros((len(neigh_range), len(k_range)))
    for i in range(len(neigh_range)):
        with open('res_KNN_model_LMNN_eculidean_neigh=' + str(neigh_range[i]) + '.txt', 'r') as f:
            lines = f.readlines()
            for j in range(len(k_range)):
                scores[i][j] = float(lines[j].split('=')[-1])
    plt.figure()
    for i in range(len(neigh_range)):
        plt.plot(k_range, scores[i], label='n_neighbors=' + str(neigh_range[i]))
    plt.xlabel('Value of K for KNN using euclidean distance')
    plt.ylabel('Testing Accuracy')
    plt.title('Accuracy on Test Data for Different K Values using euclidean distance')
    plt.legend()
    plt.savefig('knn_accuracy_LMNN.png')
    plt.show()

def singleCurvePlot():
    # 读取当前目录下名为"res_KNN_model_LMNN_eculidean_neigh=2.txt"到neigh=8的文件，为每个文件单独画一张图
    neigh_range = range(2, 9, 2)
    k_range = range(2, 20, 1)
    for i in range(len(neigh_range)):
        scores = np.zeros(len(k_range))
        with open('res_KNN_model_LMNN_eculidean_neigh=' + str(neigh_range[i]) + '.txt', 'r') as f:
            lines = f.readlines()
            for j in range(len(k_range)):
                scores[j] = float(lines[j].split('=')[-1])
        plt.figure()
        plt.plot(k_range, scores, label='n_neighbors=' + str(neigh_range[i]))
        plt.xlabel('Value of K for KNN using euclidean distance')
        plt.ylabel('Testing Accuracy')
        plt.title('Accuracy on Test Data for Different K Values using euclidean distance')
        plt.legend()
        plt.savefig('knn_accuracy_LMNN_neigh=' + str(neigh_range[i]) + '.png')
        plt.show()

if __name__ == '__main__':
    # plot()
    singleCurvePlot()