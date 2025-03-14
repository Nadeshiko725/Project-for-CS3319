import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def draw(comp_range, scores, kernel, ppl, m):
#     plt.figure()
#     plt.plot(comp_range, scores, 'bo-', linewidth=2)
#     plt.title('TSNE with SVM ' + kernel + ' kernel, perplexity=' + str(ppl))
#     plt.xlabel('n_components')
#     plt.ylabel('Accuracy')
#     plt.savefig('TSNE_' + kernel + '_' + str(ppl) + '.jpg')

def main():
    comp_range_bh = [2, 3]
    ppl_range = [10.0, 20.0, 30.0, 40.0, 50.0]
    comp_2_scores = []
    comp_3_scores = []
    for ppl in ppl_range:
        print("\nppl=%0.2d\n"%(ppl))
        # 读取文件例如res_TSNE_linear_10.0_barnes_hut.txt中的数据，得到每个ppl对应的n_comp下的acc，然后画在一张图上
        with open('res_TSNE_linear_' + str(ppl) + '_barnes_hut.txt', 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                if 'n_comp' in line and 'acc' in line:
                    # 每一行的格式是kernel: n_comp = %f, acc = %f\n，所以用空格分割后取第5个和第8个元素
                    # 第三个空格后面到第一个逗号之间是n_comp，第六个空格后面到第一个换行符之间是acc
                    comp = float(line.split(' ')[3].split(',')[0])
                    acc = float(line.split(' ')[6])
                    if comp == 2:
                        comp_2_scores.append(acc)
                    elif comp == 3:
                        comp_3_scores.append(acc)
    # 把每个ppl对应的n_comp下的acc画在一张图上
    plt.figure()
    plt.plot(ppl_range, comp_2_scores, 'bo-', linewidth=2, label='n_comp=2')
    plt.plot(ppl_range, comp_3_scores, 'ro-', linewidth=2, label='n_comp=3')
    plt.title('TSNE with SVM ' + 'linear' + ' kernel')
    plt.xlabel('perplexity')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('TSNE_' + 'linear' + '.jpg')


    
if __name__ == '__main__':
    main()