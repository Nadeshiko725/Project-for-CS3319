import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selective_search
from skimage import io, transform

Image_PATH = "AwA2-data/JPEGImages/"

def SelectiveSearch(className, imageName):
    # image = cv2.imread(Image_PATH + className + "/" + imageName + ".jpg")    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = io.imread(Image_PATH + className + "/" + imageName + ".jpg")
    # image = transform.resize(image, (224, 224))
    # 让plt适应图片大小


    boxes = selective_search.selective_search(image, mode='single', random_sort=False)
    # 输出带有box的图片
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    for box in boxes:
        rect = mpatches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    ax.set_title('Selective Search')
    plt.savefig(imageName + '_SelectiveSearch.jpg')

    boxes_filter = selective_search.box_filter(boxes, min_size=5, topN=200)
    # 输出过滤后的带有box的图片
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Filtered')
    for box in boxes_filter:
        rect = mpatches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.savefig(imageName + '_Filtered.jpg')

    boxes_fine_filter = selective_search.box_filter(boxes_filter, min_size=20, topN=60)
    # 输出最终的带有box的图片
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Fine Filtered')
    for box in boxes_fine_filter:
        rect = mpatches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.savefig(imageName + '_FineFiltered.jpg')


if __name__ == "__main__":
    # SelectiveSearch("antelope", "antelope_10001")
    SelectiveSearch("antelope", "antelope_10056")
    # SelectiveSearch("antelope", "antelope_10068")
