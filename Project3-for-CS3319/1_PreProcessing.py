import numpy as np
import pandas as pd
from requests import get
import os

Data_PATH = "AwA2-data/"

# 得到所有类别的名称以及对应的图片数量
def get_class_image_num():
    class_image_num = {}
    subdirectories = os.listdir(Data_PATH + "JPEGImages/")
    for subdirectory in subdirectories:
        class_image_num[subdirectory] = len(os.listdir(Data_PATH + "JPEGImages/" + subdirectory))
        print("Class: ", subdirectory, " Image Number: ", class_image_num[subdirectory])

    print("Total Class Number: ", len(class_image_num), " Total Image Number: ", sum(class_image_num.values())  )
    # 保存为.npy文件
    np.save("class_image_num.npy", class_image_num)

if __name__ == "__main__":
    get_class_image_num()
