import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import sys
from processData import loadSplitedData
import SVMmodel
from tqdm import tqdm


def runTSNE(X_train, X_test, y_train, y_test, comp_range, ppl, m):
    linear_scores = []
    for n_comp in tqdm(comp_range, desc="Progress"):
        print("\nn_comp=%d\n"%(n_comp))
        transformer = TSNE(n_components=n_comp, perplexity=50.0, method=m, n_jobs=8)
        transformer.fit(X_train)
        X_train_proj = transformer.fit_transform(X_train)
        X_test_proj = transformer.fit_transform(X_test)
        score_linear = SVMmodel.runSVM(X_train_proj, X_test_proj, y_train, y_test, 'linear', C = 0.001)
        linear_scores.append(score_linear.mean())
    return linear_scores

def draw(comp_range, scores, kernel, ppl, m):
    bestIdx = np.argmax(scores)
    bestNComp = comp_range[bestIdx]
    bestAcc = scores[bestIdx]
    with open('res_TSNE_' + kernel + '_' + str(ppl) + '_' + m + '.txt', 'w') as f:
        for i in range(len(comp_range)):
            f.write(kernel + ": n_comp = %f, acc = %f\n"%(comp_range[i], scores[i]))
        f.write(kernel + ": Best n_comp = %f\n"%(bestNComp))
        f.write(kernel + ": acc = %f\n"%(bestAcc))

    if m != 'barnes_hut':
        plt.figure()
        plt.plot(comp_range, scores, 'bo-', linewidth=2)
        plt.title('TSNE with SVM ' + kernel + ' kernel, perplexity=' + str(ppl))
        plt.xlabel('n_components')
        plt.ylabel('Accuracy')
        plt.savefig('TSNE_' + kernel + '_' + str(ppl) + '.jpg')

def main():
    comp_range_bh = [2, 3]
    ppl_range = [10.0, 20.0, 30.0, 40.0, 50.0]
    X_train, X_test, y_train, y_test = loadSplitedData()
    for ppl in ppl_range:
        print("\nppl=%0.2d\n"%(ppl))
        linear_scores_bh = runTSNE(X_train, X_test, y_train, y_test, comp_range_bh, ppl, 'barnes_hut')
        draw(comp_range_bh, linear_scores_bh, 'linear', ppl, 'barnes_hut')

if __name__ == '__main__':
    main()
