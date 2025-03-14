import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
label_file_name = "Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
col_name = ['feature'+str(i) for i in range(2048)]


def loadOriginalData():
    feature_data = pandas.read_csv(feature_file_name, sep=' ', header=None, names=col_name)
    label_data = pandas.read_csv(label_file_name, sep=' ', header=None, names=['label'])
    return feature_data, label_data

def splitData():
    feature_data = pandas.read_csv(feature_file_name, sep=' ', header=None, names=col_name)
    label_data = pandas.read_csv(label_file_name, sep=' ', header=None, names=['label'])
    X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.4)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

def loadSplitedData():
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    splitData()
    print('Data split done!')
