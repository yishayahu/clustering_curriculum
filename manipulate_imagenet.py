import os
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm



if __name__ == '__main__':
    name = "train"
    if name =="train":
        idx = 0
        labels = []
        for i in range(1,11):
            data_batch = load_databatch(r"C:\Users\Y\PycharmProjects\data\data_clustering\imagenet",idx=1,name=name)
            x = data_batch["X_train"]
            y = data_batch["Y_train"]
            labels.extend(list(y))
            for j in tqdm(range(x.shape[0])):
                assert y[j] == labels[idx]
                torch.save(torch.Tensor(x[j]),rf"C:\Users\Y\PycharmProjects\data\data_clustering\imagenet_by_images_32\train\image_{idx}.pt")
                idx += 1
        torch.save(torch.Tensor(labels),"labels.pt")
    else:
        idx = 0
        labels = []
        data_batch = load_databatch(r"C:\Users\Y\PycharmProjects\data\data_clustering\imagenet",idx=-1,name="val")
        x = data_batch["X_train"]
        y = data_batch["Y_train"]
        labels.extend(list(y))
        for j in tqdm(range(x.shape[0])):
            torch.save(torch.Tensor(x[j]),rf"C:\Users\Y\PycharmProjects\data\data_clustering\imagenet_by_images_32\val\image_{idx}.pt")
            idx += 1
        torch.save(torch.Tensor(labels),"labels.pt")

