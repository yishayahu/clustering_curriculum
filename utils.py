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


class Cifar10Ds(torch.utils.data.Dataset):
    def __init__(self, data_root, is_train=True, is_eval=False, max_index=5):

        self.ds = torchvision.datasets.CIFAR10(
            root=data_root, download=False, train=is_train or is_eval,
            transform=tvtf.ToTensor()  # Convert PIL image to pytorch Tensor

        )
        if is_eval and not is_train:
            self.ds = torch.utils.data.Subset(self.ds, range(40000, 50000))
        elif is_train and not is_eval and max_index == 4:
            self.ds = torch.utils.data.Subset(self.ds, range(40000))
        self.batch_len = len(self.ds)

    def restart(self):
        self.collect_garbage()

    def collect_garbage(self):
        gc.collect()

    def __getitem__(self, item):
        return self.ds[item]

    def __len__(self):
        return len(self.ds)


class DS_by_batch(torch.utils.data.Dataset):
    def __init__(self, data_root, is_train=True, is_eval=False, max_index=10):
        self.data_root = data_root

        self.data_len = 1281167  # todo:remove constant max(os.listdir(data_root), key=lambda x: int(x.split("_")[1].split(".")[0]))
        self.curr_batch_idx = 1
        if is_eval:
            assert not is_train
        if is_train:
            assert not is_eval

        self.curr_batch = None
        self.batch_len = 128116 if os.environ["my_computer"] == "False" else (
                    512 * 5)  # self.curr_batch["Y_train"].shape[0]
        if not is_train and not is_eval:
            self.batch_len = 50000 if os.environ["my_computer"] == "False" else (512 * 5)
        self.is_train = is_train
        self.is_eval = is_eval
        self.max_index = max_index if os.environ["my_computer"] == "False" else 1

    def restart(self):
        self.curr_batch_idx += 1
        if self.curr_batch_idx > self.max_index:
            self.curr_batch_idx = 1
        self.collect_garbage()

    def collect_garbage(self):
        self.curr_batch = None
        gc.collect()

    def __getitem__(self, item):
        if self.curr_batch is None:
            self.curr_batch = load_databatch(data_folder=self.data_root,
                                             idx=self.curr_batch_idx if not self.is_eval else 10,
                                             name="train" if (self.is_train or self.is_eval) else "val")
        return self.curr_batch["X_train"][item], self.curr_batch["Y_train"][item]

    def __len__(self):
        assert False
        return self.batch_len


class DS_by_image(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_len = len(os.listdir(data_root)) - 1
        self.labels = torch.load(os.path.join(data_root, "labels.pt"))
        assert self.data_len == len(self.labels)

    def __getitem__(self, item):
        im_path = os.path.join(self.data_root, f"image_{item}.pt")
        return torch.load(im_path), self.labels[item]

    def __len__(self):
        return self.data_len


def create_data_loaders(datasets, samplers):
    dls = []
    for idx in range(len(datasets)):
        if datasets[idx] != []:
            dl = torch.utils.data.DataLoader(
                datasets[idx], batch_size=int(os.environ["batch_size"]), sampler=samplers[idx],
                shuffle=True if not samplers[idx] else None,
                num_workers=0)
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

    def add_images(self, idx, images, title, step):
        img_grid = torchvision.utils.make_grid(images)
        self.writers[idx].add_image(title, img_grid, step)

    def add_scalar(self, idx, *args):
        self.writers[idx].add_scalar(*args)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, idx, img_size=32, name="train"):
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

    x = x/np.float32(255)
    if name == "train":
        mean_image = mean_image/np.float32(255)

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
        X_train=X_train,
        Y_train=Y_train)
