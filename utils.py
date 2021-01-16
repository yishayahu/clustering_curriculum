import json
import os
import hashlib
import pickle
import subprocess
import numpy as np
import torch
import torchvision
import random
import torchvision.transforms as tvtf
import gc
from augmentor import MyAugmentor
image_name_to_idx = json.load(open("image_name_to_idx.json"))

class Cifar10Ds(torch.utils.data.Dataset):
    def __init__(self, data_root, is_train=True, is_eval=False, max_index=5):
        transforms = tvtf.Compose([tvtf.RandomHorizontalFlip(p=0.5),
                                   tvtf.RandomVerticalFlip(p=0.5),
                                   tvtf.RandomRotation(degrees=(-90, 90)),
                                   tvtf.ColorJitter(brightness=0.2, contrast=0.2),
                                   tvtf.ToTensor(),
                                   ])
        self.ds = torchvision.datasets.CIFAR10(
            root=data_root, download=False, train=is_train or is_eval,
            transform=transforms if (is_train or is_eval) else tvtf.ToTensor()  # Convert PIL image to pytorch Tensor

        )
        self.base_index = 0
        if is_eval and not is_train:
            self.base_index = 40000
            self.ds = torch.utils.data.Subset(self.ds, range(40000, 50000))
        elif is_train and not is_eval and max_index == 4:
            self.ds = torch.utils.data.Subset(self.ds, range(40000))
        self.batch_len = len(self.ds)

    def restart(self):
        self.collect_garbage()

    def collect_garbage(self):
        gc.collect()

    def __getitem__(self, item):
        img, label = self.ds[item]
        return (img, item + self.base_index), label

    def __len__(self):
        return len(self.ds)


class TinyInDs(torch.utils.data.Dataset):
    def __init__(self, data_root, is_train=True, is_eval=False, max_index=500):

        if is_train or is_eval:
            transforms = tvtf.Compose([MyAugmentor(),tvtf.ToTensor()])
            data_root = os.path.join(data_root, "train")
            if is_train:

                def is_valid_file(path1):
                    return int(path1.split("_")[-1].split(".")[0]) < max_index
            else:

                assert is_eval

                def is_valid_file(path1):
                    return int(path1.split("_")[-1].split(".")[0]) >= max_index
        else:
            data_root = os.path.join(data_root, "new_val")
            transforms = tvtf.Compose([tvtf.ToTensor()])
            def is_valid_file(path1):
                return True
        self.ds = torchvision.datasets.ImageFolder(root=data_root, is_valid_file=is_valid_file,transform=transforms)
        self.batch_len  = len(self.ds) if os.environ["my_computer"] == "False" else (int(os.environ["batch_size"]) * 5)
    def __getitem__(self, item):
        img,label = self.ds[item]
        assert 0<=label<200
        image_name  = os.path.split(self.ds.imgs[item][0])[-1].split(".")[0]
        image_index = image_name_to_idx[image_name]
        return (img,image_index),label
    def __len__(self):
        len(self.ds)
    def restart(self):
        self.collect_garbage()
    def collect_garbage(self):
        gc.collect()

class DS_by_batch(torch.utils.data.Dataset):
    def __init__(self, data_root, is_train=True, is_eval=False, max_index=10):
        _MEAN = [0.485, 0.456, 0.406]
        _STD = [0.229, 0.224, 0.225]
        self.transforms = tvtf.Compose([tvtf.RandomCrop(52),
                                        tvtf.RandomVerticalFlip(p=0.25),
                                        tvtf.RandomRotation(degrees=(-90, 90)),
                                        tvtf.RandomHorizontalFlip(p=0.25)])
        self.normalize = tvtf.Normalize(_MEAN, _STD)
        self.data_root = data_root

        self.data_len = 1281167  # todo:remove constant max(os.listdir(data_root), key=lambda x: int(x.split("_")[1].split(".")[0]))
        self.is_eval = is_eval
        self.curr_batch_idx = 1
        if is_eval:
            self.curr_batch_idx = 10
            assert not is_train
        if is_train:
            assert not is_eval

        self.curr_batch = None
        self.batch_len = 128116 if os.environ["my_computer"] == "False" else (int(os.environ["batch_size"]) *5)  # self.curr_batch["Y_train"].shape[0]
        if not is_train and not is_eval:
            self.batch_len = 50000 if os.environ["my_computer"] == "False" else (int(os.environ["batch_size"]) *5)
        self.is_train = is_train
        self.max_index = max_index if os.environ["my_computer"] == "False" else 1

    def restart(self):
        if self.is_train:
            self.curr_batch_idx += 1
            if self.curr_batch_idx > self.max_index:
                self.curr_batch_idx = 1
        elif self.is_eval:
            self.curr_batch_idx = 10
        self.collect_garbage()

    def collect_garbage(self):
        self.curr_batch = None
        gc.collect()

    def __getitem__(self, item):
        if self.curr_batch is None:
            self.curr_batch = load_databatch(data_folder=self.data_root,
                                             idx=self.curr_batch_idx,
                                             name="train" if (self.is_train or self.is_eval) else "val")
        img = self.curr_batch["X_train"][item]
        if self.is_train or self.is_eval:
            img = self.transforms(img)
        raise Exception("check normalize")
        img = self.normalize(img)
        return (img, item + ((self.curr_batch_idx - 1) * self.batch_len)), self.curr_batch["Y_train"][item]

    def __len__(self):
        assert False
        return self.batch_len


def create_data_loaders(datasets, samplers):
    dls = []

    for idx in range(len(datasets)):
        if datasets[idx] != []:
            dl = torch.utils.data.DataLoader(
                datasets[idx], batch_size=int(os.environ["batch_size"]), sampler=samplers[idx],
                shuffle=True if not samplers[idx] else None,
                num_workers=2)
            dls.append(dl)
        else:
            dls.append(None)
    return tuple(dls)


def get_md5sum(bytes1):
    hasher = hashlib.md5(bytes1)

    return hasher.hexdigest()


class Tb:
    def __init__(self, exp_name):
        from torch.utils.tensorboard import SummaryWriter
        self.writers = [SummaryWriter(f'runs/{exp_name}_1'), SummaryWriter(f'runs/{exp_name}_2')]
        self.exp_name = exp_name

    def add_images(self, idx, images, title, step):
        img_grid = torchvision.utils.make_grid(images)
        self.writers[idx].add_image(title, img_grid, step)

    def add_scalar(self, idx, *args):
        self.writers[idx].add_scalar(*args)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, idx, img_size=64, name="train"):
    if name == "train":
        data_file = os.path.join(data_folder, 'train_data_batch_')
        d = unpickle(data_file + str(idx))
    else:
        data_file = os.path.join(data_folder, 'val_data')
        d = unpickle(data_file)
    x = d['data'].astype(np.float32)
    y = d['labels']
    if name == "train":
        mean_image = d['mean']

    x = x / np.float32(255)
    if name == "train":
        mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]
    # if name == "train":
    #     x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = np.array(y[0:data_size])
    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=torch.Tensor(X_train),
        Y_train=Y_train)
