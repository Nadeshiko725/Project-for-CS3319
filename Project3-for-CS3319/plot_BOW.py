import matplotlib.pyplot as plt
import numpy as np

file_name = "res_BOW_full.txt"

with open(file_name, 'r') as f:
  lines = f.readlines()

k_values = [8, 16, 32, 64, 128, 256, 512, 1024]
c_values = [0.001, 0.03, 0.1, 0.3, 1, 3, 5]
linear_scores = {}
rbf_scores = {}

# for line in lines:
#   if line.startswith('k:'):
#     # 第一个冒号后面是k，第二个冒号后面是c，分开读取
#     k = line.split()[1]
#     c = line.split()[3]

#     k_values.append(int(k))
#     c_values.append(float(c))
#   elif line.startswith('SVM model accuracy_score:'):
#     score = float(line.split()[-1])
#     if 'linearscore' in line:
#       if c not in linear_scores:
#         linear_scores[c] = []
#       linear_scores[c].append(score)
#     elif 'rbf' in line:
#       if c not in rbf_scores:
#         rbf_scores[c] = []
#       rbf_scores[c].append(score)

# 一次读入三行，第一行是k和C，第二行是linear_score，第三行是rbf_score
for i in range(0, len(lines), 3):
  print(i)
  k = lines[i].split()[1]
  c = lines[i].split()[3]
  

  linear_score = float(lines[i+1].split()[-1])
  if c not in linear_scores:
    linear_scores[c] = []
  linear_scores[c].append(linear_score)

  rbf_score = float(lines[i+2].split()[-1])
  if c not in rbf_scores:
    rbf_scores[c] = []
  rbf_scores[c].append(rbf_score)

# Plot linear scores
plt.figure(1)
for c, scores in linear_scores.items():
  plt.plot(k_values, scores, label='C = {}'.format(c))
plt.xlabel('k in BOW')
plt.ylabel('Accuracy of linear SVM model')
plt.title('Linear SVM Scores of SIFT with BOW')
plt.legend()
plt.savefig('Linear_SVM_Scores_of_SIFT_with_BOW.jpg')

# Plot rbf scores
plt.figure(2)
for c, scores in rbf_scores.items():
  plt.plot(k_values, scores, label='C = {}'.format(c))
plt.xlabel('k in BOW')
plt.ylabel('Accuracy of rbf SVM model')
plt.title('RBF SVM Scores ')
plt.legend()
plt.savefig('RBF_SVM_Scores_of_SIFT_with_BOW.jpg')

plt.show()