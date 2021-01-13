import os
use_imagenet = True
my_computer = "False"
os.environ["my_computer"] = my_computer
os.environ["batch_size"] = "512"
os.environ["use_imagenet"] = str(use_imagenet)



import torch
import torchvision
import torchvision.transforms as tvtf
import torch.nn as nn
from clustered_Sampler import ClusteredSampler, RegularSampler
import utils
from new_resnet import *
from trainer import Trainer
import clustering_algorithms
import numpy as np






def main(exp_name="cifar_10_with_aug"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print(torch.cuda.get_device_name(0))
    if str(my_computer) == "False":
        os.environ["n_cluster"] = "500" if use_imagenet else "50"
    else:
        os.environ["n_cluster"] = "10"
    print(f"n clustrs is {os.environ['n_cluster']}")

    models = [resnet50(num_classes=1000 if use_imagenet else 10,
                       clustering_algorithm=clustering_algorithms.KmeanSklearnByBatch(
                           n_clusters=int(os.environ['n_cluster'])),
                       pretrained=False),
              resnet50(num_classes=1000 if use_imagenet else 10, pretrained=False)]
    for model in models:
        model.to(device=device)
    train_dls, eval_dls, test_dls = [], [], []
    # create cluster resnet data

    if use_imagenet:
        train_set_normal, test_set = utils.DS_by_batch(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"),
            max_index=10), utils.DS_by_batch(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"), is_train=False,
            is_eval=False)
        train_set_clustered, eval_set = utils.DS_by_batch(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"),
            max_index=9), utils.DS_by_batch(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"), is_eval=True,
            is_train=False)
    else:
        train_set_normal, test_set = utils.Cifar10Ds(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering"),
            max_index=5), utils.Cifar10Ds(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering"), is_train=False,
            is_eval=False)
        train_set_clustered, eval_set = utils.Cifar10Ds(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering"),
            max_index=4), utils.Cifar10Ds(
            data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering"), is_eval=True,
            is_train=False)
    tb = utils.Tb(exp_name=exp_name)
    print("clustreee")
    if str(my_computer) == "True":
        start_clustering = 6
    elif use_imagenet:
        start_clustering = 10000
    else:
        start_clustering = 1000
    clustered_smapler = ClusteredSampler(train_set_normal, tb=tb)
    train_dl, eval_dl, test_dl = utils.create_data_loaders([train_set_clustered, eval_set, test_set],
                                                           [RegularSampler(train_set_clustered),
                                                            RegularSampler(eval_set), RegularSampler(test_set)])
    train_dls.append(train_dl)
    eval_dls.append(eval_dl)
    test_dls.append(test_dl)
    # normal resnet data
    train_dl, eval_dl, test_dl = utils.create_data_loaders([train_set_normal, [], test_set],
                                                           [RegularSampler(train_set_normal), None,
                                                            RegularSampler(test_set)])
    train_dls.append(train_dl)
    eval_dls.append(eval_dl)
    test_dls.append(test_dl)
    optimizer1 = torch.optim.SGD(models[0].parameters(), lr=0.001, momentum=0.9,nesterov=True, weight_decay=5e-4)
    scheduler1 =torch.optim.lr_scheduler.CyclicLR(optimizer1, base_lr=0.00001, max_lr=0.01,step_size_up=5000,mode="triangular2")
    optimizer2 = torch.optim.SGD(models[0].parameters(), lr=0.001, momentum=0.9,nesterov=True, weight_decay=5e-4)
    scheduler2 =torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=0.00001, max_lr=0.01,step_size_up=5000,mode="triangular2")

    trainer = Trainer(models=models, train_dls=train_dls, eval_dls=eval_dls, test_dls=test_dls,
                      loss_fn=nn.CrossEntropyLoss(), loss_fn_eval=nn.CrossEntropyLoss(reduction="none"),
                      optimizers=[optimizer1,optimizer2],schedulers=[scheduler1,scheduler2], num_steps=300000, tb=tb, load=False, clustered_sampler=clustered_smapler,
                      start_clustering=start_clustering)
    trainer.train_models()


if __name__ == '__main__':
    main()
