import json
import os
import pickle
import subprocess
import time

import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

import utils
from utils import get_md5sum, Tb
import numpy as np
import progressbar
from clustered_Sampler import RegularSampler

class Trainer:
    def __init__(self, models, train_dls, eval_dls, test_dls, loss_fn, loss_fn_eval, optimizers, num_steps, tb,load):

        self.models = models
        self.train_dls = train_dls
        self.eval_dls = eval_dls
        self.test_dls = test_dls
        self.loss_fn = loss_fn
        self.loss_fn_eval = loss_fn_eval
        self.optimizers = optimizers
        self.num_steps = num_steps
        self.curr_steps = [0] * len(models)
        self.times = {}
        self.losses = {}
        self.accuracies = {}
        self.steps_for_acc_loss_and_time = {}
        self.tb = tb
        self.clusters = None
        self.bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        self.last_bar_update = 0
        self.load = load


        for phase in ["train", "eval", "test"]:
            self.times[phase] = [[] for _ in models]
            self.losses[phase] = [[] for _ in models]
            self.accuracies[phase] = [[] for _ in models]
            self.steps_for_acc_loss_and_time[phase] = [[] for _ in models]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def load_ckpt(self,model,optimizer,state_dict):
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        return model,optimizer,state_dict["step"]


    def save_ckpt(self,model,optimizer,step,idx):
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"ckpt/model_{idx}.pth")
        if idx == 0:
            pickle.dump(model.cluster_dict,open("ckpt/cluster_dict.p","wb"))
    def run_train(self, idx):
        model = self.models[idx]
        curr_step = self.curr_steps[idx]

        dl = self.train_dls[idx]
        running_loss = 0.0
        running_corrects = 0

        optimizer = self.optimizers[idx]
        epoch_len = 0
        if self.load and curr_step == 0:
            for ckpt in os.listdir("ckpt"):
                _,ckpt_idx,_ = ckpt.split(".")[0].split("_")
                if int(ckpt_idx) == idx:
                    print(f"loading from {ckpt}")
                    state_dict = torch.load(f"ckpt/{ckpt}")
                    model,optimizer,curr_step = self.load_ckpt(model=model,optimizer=optimizer,state_dict=state_dict)#todo: fix
        model.train()
        since = time.time()
        for inputs, labels in tqdm(dl,desc=f"idx {idx}, phase train"):
            if inputs.shape[0] == 1:
                print("skipped")
                continue
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()

            # zero the parameter gradients

            # forward
            # track history if only in train
            optimizer.zero_grad()
            model.zero_grad()
            start_train = time.time()
            outputs = model(inputs)

            loss = self.loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            curr_step += 1
            epoch_len += 1
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu().item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).cpu()


            if epoch_len>2000:
                break
        if self.last_bar_update < curr_step:
            self.bar.update(curr_step)
            self.last_bar_update = curr_step
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / epoch_len * int(os.environ["batch_size"])
        epoch_acc = running_corrects.double() /  epoch_len * int(os.environ["batch_size"])
        self.save_ckpt(model=model,optimizer=optimizer,step=curr_step,idx=idx)
        return time_elapsed, epoch_loss, epoch_acc, 0
    def run_eval(self,idx):
        model = self.models[idx]
        curr_step = self.curr_steps[idx]

        model.eval()
        dl = self.eval_dls[idx]
        if  not self.train_dls[0].sampler.need_distrbition(curr_step):
            return 0, 0, 0,0
        running_loss = 0.0
        running_corrects = 0
        since = time.time()
        eval_loss_dict = {}
        optimizer = self.optimizers[idx]
        for inputs, labels in tqdm(dl,desc=f"idx {idx}, phase eval"):
            if inputs.shape[0] == 1:
                print("skipped")
                continue
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()

            # zero the parameter gradients

            # forward
            # track history if only in train
            with torch.no_grad():
                optimizer.zero_grad()
                model.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if model.do_clustering():
                    losses = self.loss_fn_eval(outputs, labels)
                    for curr_input, temp_loss in zip(inputs, losses):
                        hashed = get_md5sum(curr_input.cpu().numpy().tobytes())
                        eval_loss_dict[str(hashed)] = temp_loss
                running_loss += loss.cpu().item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu()
        if self.last_bar_update < curr_step:
            self.bar.update(curr_step)
            self.last_bar_update = curr_step
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / len(dl.dataset)
        epoch_acc = running_corrects.double() / len(dl.dataset)
        if model.do_clustering() and self.train_dls[idx].sampler.start_clustering < curr_step:
            if self.clusters is None:
                self.clusters = model.get_clusters()
            ret_value = self.train_dls[idx].sampler.create_distribiouns(self.clusters, eval_loss_dict, curr_step)  # todo: run this line only once
            if ret_value == "done":
                model.clustering_algorithm = None
                new_ds = utils.DS_by_batch(data_root=os.path.join(os.path.dirname(os.getcwd()),"data","data_clustering","imagenet"),max_index=10)
                self.train_dls[idx] = torch.utils.data.DataLoader(
                    new_ds, batch_size=int(os.environ["batch_size"]), sampler=RegularSampler(new_ds),
                    num_workers=0 if os.environ["my_computer"] == "True" else 2)
        return time_elapsed, epoch_loss, epoch_acc, 0
    def run_test(self,idx):
        model = self.models[idx]
        curr_step = self.curr_steps[idx]
        model.eval()
        dl = self.test_dls[idx]
        running_loss = 0.0
        running_corrects = 0
        sub_running_corrects = 0
        sub_running_corrects_disc = 0
        since = time.time()
        optimizer = self.optimizers[idx]
        for inputs, labels in tqdm(dl,desc=f"idx {idx}, phase test"):
            if inputs.shape[0] == 1:
                print("skipped")
                continue
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()

            with torch.no_grad():
                optimizer.zero_grad()
                model.zero_grad()
                outputs = model(inputs)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if  model.do_clustering() and self.clusters:
                    cluster_labels = model.clustering_algorithm.predict(inputs,model.cluster_dict,True)
                    is_in_train = self.train_dls[idx].sampler.is_in_train(cluster_labels)
                    assert len(is_in_train) == len(preds)
                    for pred, flag, label in zip(preds, is_in_train, labels.data):
                        if flag:
                            if pred == label:
                                sub_running_corrects += 1
                            sub_running_corrects_disc += 1
                running_loss += loss.cpu().item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu()
        if self.last_bar_update < curr_step:
            self.bar.update(curr_step)
            self.last_bar_update = curr_step
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / len(dl.dataset)
        epoch_acc = running_corrects.double() / len(dl.dataset)
        if sub_running_corrects_disc == 0:
            sub_epoch_acc = None
        else:
            sub_epoch_acc = sub_running_corrects / sub_running_corrects_disc

        return time_elapsed, epoch_loss, epoch_acc, sub_epoch_acc


    def save_epoch_results(self, phase, idx, curr_time, loss, acc, sub_acc, step):
        def save_results_to_json():
            json.dump(self.times, open("times.json", 'w'))
            json.dump(self.losses, open("losses.json", 'w'))
            json.dump(self.accuracies, open("accuracies.json", 'w'))
            json.dump(self.steps_for_acc_loss_and_time, open("steps.json", 'w'))

        def save_results_to_tb():
            self.tb.add_scalar(idx, phase + " loss", loss, step)
            self.tb.add_scalar(idx, phase + " curr_time", curr_time, step)
            self.tb.add_scalar(idx, phase + " acc", acc, step)
            if sub_acc:
                self.tb.add_scalar(idx, phase + " sub_acc", sub_acc, step)

        self.times[phase][idx].append(curr_time)
        self.losses[phase][idx].append(loss)
        self.accuracies[phase][idx].append(acc.item())
        self.steps_for_acc_loss_and_time[phase][idx].append(self.curr_steps[idx])
        save_results_to_json()
        save_results_to_tb()

    def train_models(self):
        while True:
            if min(self.curr_steps) > self.num_steps:
                break
            idx = np.argmin(self.curr_steps)
            assert idx in [0, 1]  # todo: remove
            curr_time, loss, acc, sub_acc  = self.run_train(idx)
            if (curr_time, loss, acc, sub_acc) != (0,0,0,0):

                self.save_epoch_results("train", idx, curr_time, loss, acc, sub_acc, self.curr_steps[idx])
            if self.models[idx].do_clustering():
                curr_time, loss, acc, sub_acc = self.run_eval(idx)
                if (curr_time, loss, acc, sub_acc) != (0,0,0,0):
                    self.save_epoch_results("eval", idx, curr_time, loss, acc, sub_acc, self.curr_steps[idx])
            curr_time, loss, acc, sub_acc  = self.run_test(idx)
            if (curr_time, loss, acc, sub_acc) != (0,0,0,0):
                self.save_epoch_results("test", idx, curr_time, loss, acc, sub_acc, self.curr_steps[idx])



