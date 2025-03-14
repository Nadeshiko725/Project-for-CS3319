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
from multiprocessing import Pool, cpu_count, Manager, Process, Queue
import time
import tqdm
import pickle
import os

SIFT_PATH = "AwA2-data/SIFT_LD/"
DL_PATH = "AwA2-data/DL_LD/"
y_file_name = "AwA2-data/AwA2-labels.txt"

class_image_num = np.load("class_image_num.npy", allow_pickle=True).item()
local_descripetor = np.load("local_descripetor.npy", allow_pickle=True)
class_num_group = np.load("class_num_group.npy", allow_pickle=True)
local_descripetor_less = np.load("local_descripetor_less.npy", allow_pickle=True)
class_num_group_less = np.load("class_num_group_less.npy", allow_pickle=True)

class VLADProcess(Process):
    def __init__(self, class_num_group, k, model, q, index):
        super(VLADProcess, self).__init__()
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
                vlad = [np.zeros((1, local_descripetor.shape[1])) for i in range(self.k)]
                for descriptor in tmp:
                    label = self.model.predict(descriptor.reshape(1, -1))[0]
                    vlad[label] += descriptor - self.model.cluster_centers_[label]
                vlad = np.hstack(vlad)
                feature.append(vlad)
        self.q.put((self.index, np.vstack(feature)))

def get_VLAD_feature(k):
    model = KMeans(n_clusters=k, copy_x=False, n_init='auto')
    model.fit(local_descripetor)

    q = Queue()
    feature = [None for i in range(8)]
    processPool = [VLADProcess(class_num_group_less[i], k, model, q, i) for i in range(8)]
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
    # k_range = [8, 16, 32, 64, 128, 256, 512, 1024]
    k_range = [512, 1024]
    C_range = [0.001, 0.03, 0.1, 0.3, 1, 3, 5]


    for k in tqdm.tqdm(k_range, colour='blue'):
        if(os.path.exists("VLAD_feature_" + str(k) + ".pkl")):
            with open("VLAD_feature_" + str(k) + ".pkl", 'rb') as f:
                X = pickle.load(f)
        else:
            print("starting get VLAD feature")
            X = get_VLAD_feature(k)
            with open("VLAD_feature_" + str(k) + ".pkl", 'wb') as f:
                pickle.dump(X, f)
        print("get VLAD feature done")

        pca = PCA(n_components=49)
        X = pca.fit_transform(X)
        print("PCA done")
        col_name = ['feature_' + str(i) for i in range(X.shape[1])]
        X_Scaled = StandardScaler().fit_transform(X)
        X_Scaled = pd.DataFrame(data=X_Scaled, columns=col_name)
        y = fitYwithX(X.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, test_size=0.4, random_state=42)
        print("split done")

        for C in tqdm.tqdm(C_range, colour='red'):
            # 如果k=512，C从3开始迭代
            if(k == 512 and C < 3):
                continue
            print("k: ", k, " C: ", C)
            linear_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, 'linear', C)
            rbf_score = SVMmodel.runSVM(X_train, X_test, y_train, y_test, 'rbf', C)
            with open('res_VLAD_less.txt', 'a') as f:
                f.write("VLAD encoding: k = %d, C = %f, linear_score = %f, rbf_score = %f\n"%(k, C, linear_score, rbf_score))

if __name__ == "__main__":
    start = time.time()
    main()
    print("Time: ", time.time() - start)