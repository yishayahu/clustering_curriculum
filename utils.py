import os
import hashlib
import subprocess
import numpy as np
import torch
import torchvision
import random

class DS(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.batch_len = 128116
        self.data_len =1281167# todo:remove constant max(os.listdir(data_root), key=lambda x: int(x.split("_")[1].split(".")[0]))
        self.buffer = {}

    def __getitem__(self, item):
        batch_num = int(item/self.batch_len)+1
        place_in_batch = item % self.batch_len
        if batch_num > 10:
            batch_num = 10
            place_in_batch = item - 9 * self.batch_len
        if batch_num in self.buffer:
            batch = self.buffer[batch_num]
        else:
            batch = np.load(os.path.join(self.data_root,f"train_data_batch_{batch_num}"),allow_pickle=True)
            self.buffer[batch_num] = batch
            if len(self.buffer) > 3:
                self.buffer.pop(random.choice(list(self.buffer.keys())))
        data = torch.Tensor(batch["data"][place_in_batch].reshape((32, 32, 3)).transpose(2, 0, 1))
        label = batch["labels"][place_in_batch] -1

        return data,label

    def __len__(self):
        return self.data_len


def create_data_loaders(datasets, samplers):
    dls = []
    for idx in range(len(datasets)):
        if datasets[idx]:
            dl = torch.utils.data.DataLoader(
                datasets[idx], batch_size=32, sampler=samplers[idx], shuffle=True if not samplers[idx] else None,
                num_workers=0 if os.environ["my_computer"] == "True" else 2)
            dls.append(dl)
        else:
            dls.append(None)
    return tuple(dls)


def get_md5sum(bytes1):
    hasher = hashlib.md5(bytes1)
    # hasher.update(bytes1)
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
