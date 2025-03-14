from tracemalloc import start
from BOW import SIFT_PATH_LESS
from click import Path
from cv2 import SIFT
import numpy as np
import tqdm
import time
import gc

class_image_num = np.load('class_image_num.npy', allow_pickle=True).item()
SIFT_PATH = "AwA2-data/SIFT_LD/"
SIFT_PATH_LESS = "AwA2-data/SIFT_LD_LESS/"

def get_shuffled_LD(className, totalNum, Path, proportion):
    totalLD = np.load(Path + className +'/'+ className+ "_10001.npy", allow_pickle=True)
    for i in tqdm.tqdm(range(10002, 10001 + totalNum), colour='green'):
        tmp = np.load(Path + className +'/'+ className+ "_"+str(i)+".npy", allow_pickle=True)
        # print("Processing: ", className, " Num: ", i, " Shape: ", tmp.shape)
        if(tmp.shape.__len__() != 2 or tmp.shape[1] != 128):
            print("Error: ", className, " Num: ", i, " Shape: ", tmp.shape)
            continue
        totalLD = np.concatenate((totalLD, tmp), axis=0) # 数组均为128列
    random_array = np.arange(totalLD.shape[0])
    np.random.shuffle(random_array)
    num = totalLD.shape[0] // proportion
    return totalLD[random_array[:num]]
 
def get_class_shuffled(Path, propotion):
    local_descripetor = []
    for className, totalNum in class_image_num.items():
        print("Processing: ", className, " TotalNum: ", totalNum)
        local_descripetor.append(get_shuffled_LD(className, totalNum, Path, propotion))
    return np.vstack(local_descripetor)

def get_class_shuffled_less(Path, propotion):
    local_descripetor = []
    for className, totalNum in class_image_num.items():
        print("Processing: ", className, " TotalNum: ", 100)
        local_descripetor.append(get_shuffled_LD(className, 100, Path, propotion))
    return np.vstack(local_descripetor)

def main():
    # Path = SIFT_PATH
    Path = SIFT_PATH_LESS
    start = time.time()
    propotion = 10
    # local_descripetor = get_class_shuffled(Path, propotion)
    local_descripetor = get_class_shuffled_less(Path, propotion)
    print("Time: ", time.time() - start)
    np.save("local_descripetor_less.npy", local_descripetor)

def divideClassNum():
    # 把class_image_num分成8组，便于多进程处理
    class_num_group = [{} for i in range(8)]
    i = 0
    for className, totalNum in class_image_num.items():
        # class_num_group[i % 8][className] = totalNum
        class_num_group[i % 8][className] = 100
        i += 1
    # np.save("class_num_group.npy", class_num_group)
    np.save("class_num_group_less.npy", class_num_group)


if __name__ == "__main__":
    main()
    divideClassNum()
    gc.collect()