import matplotlib.pyplot as plt
import numpy as np

file_path = "res_txt/res_VLAD_less.txt"

def plot_BOW(file_name):
    # 读取每一行的k和c以及对应的linear_score和rbf_score，分别存储
    k_values = [8, 16, 32, 64, 128, 256, 512, 1024]
    c_values = [0.001, 0.03, 0.1, 0.3, 1, 3, 5]
    linear_scores = {}
    rbf_scores = {}
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)): 
        print(i)
        c = float(lines[i].split()[7].replace(',', ''))
        linear_score = float(lines[i].split()[10].replace(',', ''))
        rbf_score = float(lines[i].split()[13].replace(',', ''))
        if c not in linear_scores:
            linear_scores[c] = []
        if c not in rbf_scores:
            rbf_scores[c] = []
        linear_scores[c].append(linear_score)
        rbf_scores[c].append(rbf_score)

    # 横坐标为k，纵坐标为accuracy，用不同的颜色代表不同的c值，用实线代表linear_score，用虚线代表rbf_score，把两种score画在一张图上
    plt.figure()
    for c, scores in linear_scores.items():
        plt.plot(k_values, scores, label='C = {}'.format(c))
    for c, scores in rbf_scores.items():
        plt.plot(k_values, scores, label='C = {}'.format(c), linestyle='--')
    plt.xlabel('k in VLAD')
    plt.ylabel('Accuracy of linear and rbf SVM model')
    plt.title('Scores of SIFT with VLAD based on Randm Sampled Dataset')
    plt.legend(bbox_to_anchor=(0.85, 0.95   ), loc='upper left')
    plt.savefig('Scores_of_SIFT_with_VLAD_LESS.jpg')
    plt.show()
    
    
if __name__ == "__main__":
    plot_BOW(file_path)