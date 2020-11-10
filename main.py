import os
import torch
import torchvision
import torchvision.transforms as tvtf
from sklearn.cluster import KMeans
import torch.nn as nn
from kmeanstf import KMeansTF

from clustered_Sampler import ClusteredSampler
import utils
from new_resnet import resnet50
from trainer import Trainer
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    my_computer = str(device) == "cpu"
    os.environ["my_computer"] = str(my_computer)
    print(str(my_computer))
    if str(my_computer) == "False":
        os.system("pip install ImageHash")
        os.environ["n_cluster"] = "40"
    else:
        os.environ["n_cluster"] = "10"
    print(f"n clustrs is {os.environ['n_cluster']}")
    cfar10_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    cifar10_train_ds = torchvision.datasets.CIFAR10(
        root='./data/cifar-10/', download=True, train=True,
        transform=tvtf.ToTensor()  # Convert PIL image to pytorch Tensor

    )
    if os.environ["my_computer"] == "True":
        evens = list(range(0, len(cifar10_train_ds), 100))
        cifar10_train_ds = torch.utils.data.Subset(cifar10_train_ds, evens)
    models = [resnet50(num_classes=10,clustering_algorithm=KMeansTF(n_clusters=int(os.environ['n_cluster']), n_init=2, max_iter=50)),resnet50(num_classes=10)]
    for model in models:
        model.to(device=device)
    train_dls,eval_dls,test_dls = [],[],[]
    # create cluster resnet data
    train_set_normal,test_set = torch.utils.data.random_split(cifar10_train_ds, [int(len(cifar10_train_ds) * 0.85), int(len(cifar10_train_ds) * 0.15)])
    train_set_clustered,eval_set = torch.utils.data.random_split(train_set_normal, [int(len(train_set_normal) * 0.80), int(len(train_set_normal) * 0.20)])
    clustered_smapler = ClusteredSampler(eval_set,start_clustering=0, end_clustering=25000)
    train_dl, eval_dl,test_dl = utils.create_data_loaders([train_set_clustered, eval_set,test_set],[clustered_smapler,None,None])
    train_dls.append(train_dl)
    eval_dls.append(eval_dl)
    test_dls.append(test_dl)
    # normal resnet data
    train_dl, eval_dl,test_dl = utils.create_data_loaders([train_set_normal, [],test_set],[clustered_smapler,None,None])
    train_dls.append(train_dl)
    eval_dls.append(eval_dl)
    test_dls.append(test_dl)
    optimizers = [torch.optim.Adam(models[0].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False),
                  torch.optim.Adam(models[1].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)]
    trainer = Trainer(models=models,train_dls = train_dls,eval_dls=eval_dls,test_dls=test_dls,
                      loss_fn= nn.CrossEntropyLoss(),loss_fn_eval=nn.CrossEntropyLoss(reduction="none"),
                      optimizers=optimizers, num_steps=125000)
    trainer.train_models()



if __name__ == '__main__':
    main()

