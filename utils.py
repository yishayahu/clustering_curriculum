import os

import torch


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


