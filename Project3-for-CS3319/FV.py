from modulefinder import packagePathMap
from threading import local
from tracemalloc import start
from cv2 import SIFT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BOW import SIFT_PATH_LESS
import SVMmodel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool, cpu_count, Manager, Process, Queue
import time
import tqdm
import pickle
import os
import math

SIFT_PATH = "AwA2-data/SIFT_LD/"
SIFT_PATH_LESS = "AwA2-data/SIFT_LD_less/"
DL_PATH = "AwA2-data/DL_LD/"
y_file_name = "AwA2-data/AwA2-labels.txt"

class_image_num = np.load("class_image_num.npy", allow_pickle=True).item()
local_descripetor = np.load("local_descripetor.npy", allow_pickle=True)
class_num_group = np.load("class_num_group.npy", allow_pickle=True)
local_descripetor_less = np.load("local_descripetor_less.npy", allow_pickle=True)
class_num_group_less = np.load("class_num_group_less.npy", allow_pickle=True)

class FVProcess(Process):
    def __init__(self, class_num_group, k, model, q, index):
        super(FVProcess, self).__init__()
        self.class_num_group = class_num_group
        self.k = k
        self.model = model
        self.q = q
        self.index = index

    def run(self):
        feature = []
        for className, totalNum in tqdm.tqdm(self.class_num_group.items(), colour='green'):
            print("Processing: ", className, " TotalNum: ", totalNum)
            for index in range(10001, 10001 + totalNum):
                tmp = np.load(SIFT_PATH_LESS + className + '/' + className + "_" + str(index) + ".npy", allow_pickle=True)
                if(tmp.shape.__len__() != 2 or tmp.shape[1] != 128):
                    print("Error: ", className, " Num: ", index, " Shape: ", tmp.shape)
                    continue
                fv = [np.zeros((1, tmp.shape[1])) for i in range(2 * self.k)]
                for i in range(self.k):
                    for descriptor in tmp:
                        gamma = self.model.predict_proba(descriptor.reshape(1, -1))[0][i]
                        mu = self.model.means_[i]
                        sigma = np.diagonal(self.model.covariances_[i])
                        pi = self.model.weights_[i]
                        fv[i * 2] += gamma * (descriptor - mu) / sigma
                        fv[i * 2 + 1] += gamma * (np.square(descriptor - mu) / np.square(sigma) - 1)
                    fv[i * 2] /= tmp.shape[0] * math.sqrt(pi)
                    fv[i * 2 + 1] /= tmp.shape[0] * math.sqrt(2 * pi)
                fv = np.hstack(fv)
                feature.append(fv)
        self.q.put((self.index, np.vstack(feature)))

def get_FV_feature(k):
    print("starting GMM")
    # 如果有对应的pickle文件，就直接读取
    if os.path.exists('GMM_' + str(k) + '.pkl'):
        with open('GMM_' + str(k) + '.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        model = GaussianMixture(n_components=k, max_iter=200)
        model.fit(local_descripetor_less)
        # 保存fit好的model
        with open('GMM_' + str(k) + '.pkl', 'wb') as f:
            pickle.dump(model, f)
    print("GMM done")

    q = Queue()
    feature = [None for i in range(8)]
    processPool = [FVProcess(class_num_group_less[i], k, model, q, i) for i in range(8)]
    for p in processPool:
        p.start()
    for i in range(8):
        index, data = q.get()
        feature[index] = data
    for p in processPool:
        p.join()
    return np.vstack(feature)

def fitYwithX(num):
    y = pd.read_csv(y_file_name, names=['label'])
    # X只取了每个class的前100个图片，所以y也只取前100个
    return y[:num]

def main():
    # k_range = [2, 4, 8, 16, 32, 64, 128, 256]
    # k_range = [8, 16, 32, 64, 128, 256]
    k_range = [32, 64, 128, 256]
    C_range = [0.001, 0.03, 0.1, 0.3, 1, 3, 5]

    for k in tqdm.tqdm(k_range, colour='blue'):
        # 如果已经有对应的pickle文件，就直接读取
        if os.path.exists('FV_' + str(k) + '.pkl'):
            with open('FV_' + str(k) + '.pkl', 'rb') as f:
                X = pickle.load(f)
        else:
            print("starting get FV feature")
            X = get_FV_feature(k)
            with open('FV_' + str(k) + '.pkl', 'wb') as f:
                pickle.dump(X, f)
        pca = PCA(n_components=49)
        X = pca.fit_transform(X)
        print("PCA done")
        col_name = ['feature_' + str(i) for i in range(X.shape[1])]
        X_Scaled = StandardScaler().fit_transform(X)
        X_Scaled = pd.DataFrame(data=X_Scaled, columns=col_name)
        y = fitYwithX(X.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        print("split done")

        for C in tqdm.tqdm(C_range, colour='red'):
            print("k: ", k, " C: ", C)
            linear_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, 'linear', C)
            rbf_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, 'rbf', C)
            with open('res_FV_' + str(k) + '_PCA_.txt', 'a') as f:
                f.write('k: %d, C: %f, linear_score: %f, rbf_score: %f\n'%(k, C, linear_score, rbf_score))
        
if __name__ == '__main__':
    start = time.time()
    main()
    print("Time: ", time.time() - start)

        