import os
import hashlib
import torch
import torchvision


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
    def __init__(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writers = [SummaryWriter('runs/model_1'),SummaryWriter('runs/model_2')]
    def add_images(self,images):
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image("batch",img_grid)
    def add_scalar(self,idx,*args):
        self.writers[idx].add_scalar(*args)

