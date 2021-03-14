import os
my_computer = "False"
os.environ["my_computer"] = my_computer
os.environ["batch_size"] = "256" if my_computer == "False" else "26"
os.environ["dataset_name"] = "tiny_imagenet"
os.environ['PYTHONHASHSEED'] = str(101)
network_to_use = "DenseNet"
# network_to_use = "ResNet50"
optimizer_to_use = ""
schduler_to_use = ""

import torch
import torchvision
import torchvision.transforms as tvtf
import torch.nn as nn
from scheduler_wrapper import chainedCyclicLr
from clustered_Sampler import ClusteredSampler, RegularSampler
import utils
from new_resnet import *
from new_densenet import DenseNet
from trainer import Trainer
import clustering_algorithms
import numpy as np
def seed_everything():
    seed= int(os.environ['PYTHONHASHSEED'])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(exp_name="eval_at_the_end_and_viz",load=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    seed_everything()
    print(torch.cuda.get_device_name(0))
    torch.seed()
    if str(my_computer) == "False":
        n_clusters = None
        if os.environ["dataset_name"] == "imagenet":
            n_clusters = 512
            os.environ["batch_size"] = 512
        elif os.environ["dataset_name"] == "cifar10":
            n_clusters = 20
        elif os.environ["dataset_name"] == "tiny_imagenet":
            n_clusters = 200
        os.environ["n_cluster"] = str(n_clusters)
    else:
        os.environ["n_cluster"] = "10"
    print(f"n clustrs is {os.environ['n_cluster']}")
    print(f"batch size is {os.environ['batch_size']}")
    n_classes = None
    if os.environ["dataset_name"] == "imagenet":
        n_classes = 1000
    elif os.environ["dataset_name"] == "cifar10":
        n_classes = 10
    elif os.environ["dataset_name"] == "tiny_imagenet":
        n_classes = 200

    if network_to_use == "DenseNet":
        models = [DenseNet(200,clustering_algorithm=clustering_algorithms.KmeanSklearnByBatch(
            n_clusters=int(os.environ['n_cluster']))),DenseNet(200)]
        # models = [DenseNet(200),resnet50(num_classes=200,pretrained=False)]
        base_lrs = [0.0001,0.00001,0.00001,0.000001]
        max_lrs = [0.0006,0.00006,0.00006,0.000006]
        step_sizes_up = [4686,4686,3128,1564]
        ths = [0.52,0.61,0.62,0.99]
        optimizer1 = torch.optim.RMSprop(models[0].parameters(), lr=0.0001, eps=1e-08)
        scheduler1 = chainedCyclicLr(optimizer=optimizer1,base_lrs=base_lrs,max_lrs=max_lrs,step_sizes_up=step_sizes_up,ths=ths)
        optimizer2 = torch.optim.RMSprop(models[1].parameters(), lr=0.0001, eps=1e-08)
        scheduler2 = chainedCyclicLr(optimizer=optimizer2,base_lrs=base_lrs,max_lrs=max_lrs,step_sizes_up=step_sizes_up,ths=ths)
    elif network_to_use == "ResNet50":
        models = [resnet50(num_classes=n_classes,
                           clustering_algorithm=clustering_algorithms.KmeanSklearnByBatch(
                               n_clusters=int(os.environ['n_cluster'])),
                           pretrained=False),
                  resnet50(num_classes=n_classes, pretrained=False)]
        optimizer1 = torch.optim.SGD(models[0].parameters(), lr=0.001, momentum=0.9,nesterov=True, weight_decay=5e-4)
        scheduler1 =torch.optim.lr_scheduler.CyclicLR(optimizer1, base_lr=0.00001, max_lr=0.01,step_size_up=5000,mode="triangular2")
        optimizer2 = torch.optim.SGD(models[1].parameters(), lr=0.001, momentum=0.9,nesterov=True, weight_decay=5e-4)
        scheduler2 =torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=0.00001, max_lr=0.01,step_size_up=5000,mode="triangular2")
    else:
        models = []

    for model_idx, model in enumerate(models):
        print(f"copy model {model_idx} to device")
        # print(model)
        # print(f"model {model_idx} total param is {sum(p.numel() for p in model.parameters())}")
        # print(f"model {model_idx} traineble param is {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model.to(device=device)
        # print(os.popen('nvidia-smi').read())
    train_dls, eval_dls, test_dls = [], [], []
    # create cluster resnet data

    if os.environ["dataset_name"] == "imagenet":

        data_root = "/home/ML_courses/datasets/imagenet/"
        train_set_normal, test_set = utils.ImageNetDs(
            data_root=data_root,
            max_index=500, do_aug=True), utils.ImageNetDs(
            data_root=data_root, is_train=False,
            is_eval=False, do_aug=False)
        train_set_clustered, eval_set = utils.ImageNetDs(
            data_root=data_root,
            max_index=400, do_aug=False), utils.ImageNetDs(
            data_root=data_root, is_eval=True,
            is_train=False, max_index=400, do_aug=False)
        # train_set_normal, test_set = utils.DS_by_batch(
        #     data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"),
        #     max_index=10), utils.DS_by_batch(
        #     data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"), is_train=False,
        #     is_eval=False)
        # train_set_clustered, eval_set = utils.DS_by_batch(
        #     data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"),
        #     max_index=9), utils.DS_by_batch(
        #     data_root=os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering", "imagenet"), is_eval=True,
        #     is_train=False)

    elif os.environ["dataset_name"] == "cifar10":
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
    elif os.environ["dataset_name"] == "tiny_imagenet":
        data_root = "/content/tiny-imagenet-200" if my_computer == "False" else os.path.join(os.path.dirname(os.getcwd()), "data", "data_clustering","tiny-imagenet-200")
        train_set_normal, test_set = utils.TinyInDs(
            data_root=data_root,
            max_index=500,do_aug=True), utils.TinyInDs(
            data_root=data_root, is_train=False,
            is_eval=False,do_aug=False)
        train_set_clustered, eval_set = utils.TinyInDs(
            data_root=data_root,
            max_index=400,do_aug=False), utils.TinyInDs(
            data_root=data_root, is_eval=True,
            is_train=False,max_index=400,do_aug=False)
    else:
        raise Exception("1")
    tb = utils.Tb(exp_name=exp_name)
    print("clustreee")
    if str(my_computer) == "True":
        start_clustering = 6
    else:
        if os.environ["dataset_name"] == "imagenet":
            start_clustering = 20000
        elif os.environ["dataset_name"] == "cifar10":
            start_clustering = 1000
        elif os.environ["dataset_name"] == "tiny_imagenet":
            start_clustering = 3000

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


    trainer = Trainer(models=models, train_dls=train_dls, eval_dls=eval_dls, test_dls=test_dls,
                      loss_fn=nn.CrossEntropyLoss(), loss_fn_eval=nn.CrossEntropyLoss(reduction="none"),
                      optimizers=[optimizer1,optimizer2],schedulers=[scheduler1,scheduler2], num_steps=300000, tb=tb, load=load, clustered_sampler=clustered_smapler,
                      start_clustering=start_clustering)
    trainer.train_models()


if __name__ == '__main__':
    main()
