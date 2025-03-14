import numpy as np
import pandas as pd
import cv2
import tqdm
import os

Image_PATH = "AwA2-data/JPEGImages/"
SIFT_PATH = "AwA2-data/SIFT_LD/"
SIFT_PATH_LESS = "AwA2-data/SIFT_LD_LESS/"

def get_sift_features(class_name, image_name):
    image = cv2.imread(Image_PATH + class_name + "/" + image_name + ".jpg")
    image = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    KeyPoints, Descriptor = sift.detectAndCompute(gray, None)
    return KeyPoints, Descriptor

def save_sift_features():
    class_image_num = np.load("class_image_num.npy", allow_pickle=True).item()
    for className, totalNum in tqdm.tqdm(class_image_num.items(), colour="green"):
        print("Class: ", className, " Total Number: ", totalNum)
        # 创建文件夹
        os.makedirs(SIFT_PATH_LESS + className, exist_ok=True)
        for i in tqdm.tqdm(range(10001, 10101), colour="red"):
            image_name = className + "_" + str(i)
            KeyPoints, Descriptor = get_sift_features(className, image_name)
            # np.save(SIFT_PATH_LESS + className + "/" + image_name + "_KeyPoints.npy", KeyPoints)
            np.save(SIFT_PATH_LESS + className + "/" + image_name, Descriptor)
        print("Class: ", className, " Done!")

# 增强提取
def enhance_process(class_name, image_name):
    image = cv2.imread(Image_PATH + class_name + "/" + image_name + ".jpg")
    image = cv2.resize(image, (224, 224))
    # 提高图像的对比度来提取图像的局部描述符
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_dst = clahe.apply(gray) # 必须单通道否侧报错
    sift = cv2.SIFT_create()
    KeyPoints, Descriptor = sift.detectAndCompute(img_dst, None)
    return KeyPoints, Descriptor

if __name__ == "__main__":
    save_sift_features()
    
    KeyPoints, Descriptor = enhance_process("dolphin", "dolphin_10180")
    print("Descriptor Shape: ", Descriptor.shape)
    np.save(SIFT_PATH_LESS + "dolphin/dolphin_10180.npy", Descriptor)

    KeyPoints, Descriptor = enhance_process("humpback+whale", "humpback+whale_10428")
    print("Descriptor Shape: ", Descriptor.shape)
    np.save(SIFT_PATH_LESS + "humpback+whale/humpback+whale_10428.npy", Descriptor)
