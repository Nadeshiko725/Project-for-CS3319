from tkinter import image_names
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import cv2

Image_PATH = "AwA2-data/JPEGImages/dolphin/"

# imageName = "../AwA2-data/JPEGImages/antelope/antelope_10001.jpg"
# imageName = "../ImgChanged/dolphin_10180.jpg"
# imageName = "../ImgChanged/humpback+whale_10428.jpg"
imageName = "dolphin_10180.jpg"

def drawDemo(imageName):
    image = cv2.imread(Image_PATH + imageName)
    image = cv2.resize(image, (224, 224))
    cv2.imwrite("origin.jpg", image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_dst = clahe.apply(gray_image)
    cv2.imwrite("enhanced.jpg", img_dst)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_dst, None)
    # result = cv2.drawKeypoints(gray_image, kp, None)
    result = cv2.drawKeypoints(img_dst, kp, None)
    cv2.imwrite("demo.jpg", result)

if __name__ == '__main__':
    drawDemo(imageName)
