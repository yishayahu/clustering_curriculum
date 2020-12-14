import os
import hashlib
import subprocess
import numpy as np
import torch
import torchvision


class DS(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_len =1281168# todo:remove constant max(os.listdir(data_root), key=lambda x: int(x.split("_")[1].split(".")[0]))

    def __getitem__(self, item):
        im_path = os.path.join(self.data_root, f"image_{item}.npz")
        np_item = np.load(im_path)
        return torch.Tensor(np_item["data"].reshape((32, 32, 3)).transpose(2, 0, 1)), np_item["label"] -1

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
