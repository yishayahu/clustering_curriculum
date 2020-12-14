import os
import torch
import torchvision
import torchvision.transforms as tvtf

import torch.nn as nn

from clustered_Sampler import ClusteredSampler
import utils
from new_resnet import *
from trainer import Trainer
import clustering_algorithms
import numpy as np

def main(exp_name="not_pretrained_start_from_easy"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    my_computer = str(device) == "cpu"
    os.environ["my_computer"] = str(my_computer)
    if str(my_computer) == "False":
        os.environ["n_cluster"] = "4000"
    else:
        os.environ["n_cluster"] = "10"
    print(f"n clustrs is {os.environ['n_cluster']}")
    torch_ds_train = utils.DS(data_root=os.path.join(os.path.dirname(os.getcwd()),"data","data_clustering","imagenet"))
    eval_np = np.load(os.path.join(os.path.dirname(os.getcwd()),"data","data_clustering","imagenet","val_data"),allow_pickle=True)
    torch_ds_eval = torch.utils.data.TensorDataset(torch.Tensor(eval_np["data"]),torch.Tensor(eval_np["labels"]))
    models = [resnet50(num_classes=1000,
                       clustering_algorithm=clustering_algorithms.KmeanSklearn(n_clusters=int(os.environ['n_cluster'])),
                       pretrained=False),
              resnet50(num_classes=1000, pretrained=False)]
    for model in models:
        model.to(device=device)
    train_dls, eval_dls, test_dls = [], [], []
    # create cluster resnet data
    train_set_normal, test_set = torch_ds_train,torch_ds_eval
    adder = 0
    if int(len(train_set_normal) * 0.80) + int(len(train_set_normal) * 0.20) < len(train_set_normal):
        adder = 1
    train_set_clustered, eval_set = torch.utils.data.random_split(train_set_normal, [int(len(train_set_normal) * 0.80)+adder,
                                                                                     int(len(train_set_normal) * 0.20)])
    tb = utils.Tb(exp_name=exp_name)
    print("clustreee")
    clustered_smapler = ClusteredSampler(train_set_clustered, start_clustering=30000, end_clustering=250000, tb=tb)
    train_dl, eval_dl, test_dl = utils.create_data_loaders([train_set_clustered, eval_set, test_set],
                                                           [clustered_smapler, None, None])
    train_dls.append(train_dl)
    eval_dls.append(eval_dl)
    test_dls.append(test_dl)
    # normal resnet data
    train_dl, eval_dl, test_dl = utils.create_data_loaders([train_set_clustered, [], test_set], [None, None, None])
    train_dls.append(train_dl)
    eval_dls.append(eval_dl)
    test_dls.append(test_dl)
    optimizers = [torch.optim.Adam(models[0].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                   amsgrad=False),
                  torch.optim.Adam(models[1].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                   amsgrad=False)]
    trainer = Trainer(models=models, train_dls=train_dls, eval_dls=eval_dls, test_dls=test_dls,
                      loss_fn=nn.CrossEntropyLoss(), loss_fn_eval=nn.CrossEntropyLoss(reduction="none"),
                      optimizers=optimizers, num_steps=300000, tb=tb)
    trainer.train_models()


if __name__ == '__main__':
    main()
