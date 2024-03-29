import gc
import json
import os
import pickle
import subprocess
import time

import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

import utils
from label_to_str import label_to_str

import numpy as np
import progressbar
from clustered_Sampler import RegularSampler


class Trainer:
    def __init__(self, models, train_dls, eval_dls, test_dls, loss_fn, loss_fn_eval, optimizers,schedulers, num_steps, tb, load,
                 clustered_sampler, start_clustering):
        self.models = models
        self.train_dls = train_dls
        self.eval_dls = eval_dls
        self.test_dls = test_dls
        self.loss_fn = loss_fn
        self.loss_fn_eval = loss_fn_eval
        self.optimizers = optimizers
        self.schedulers = schedulers
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
        self.last_save = [0,0]
        if load:
            print("loding from ckpt")
            for i in [0, 1]:
                state_dict = torch.load(f"ckpt/model_{self.tb.exp_name}_{i}.pth")
                self.models[i].load_state_dict(state_dict["model_state_dict"])
                self.optimizers[i].load_state_dict(state_dict["optimizer_state_dict"])
                self.schedulers[i].load_state_dict(state_dict["sched_state_cidt"])
                self.curr_steps[i] = state_dict["step"]
                self.last_save[i] = state_dict["step"]
                if i == 0:
                    if self.curr_steps[0] > start_clustering:
                        pkl_filename = f"ckpt/{self.tb.exp_name}_sampler_cluster_dict.pkl"
                        with open(pkl_filename, 'rb') as file:
                            clustered_sampler.cluster_dict = pickle.load(file)
                        clustered_sampler.center = state_dict["center"]
                        clustered_sampler.hiererchy = state_dict["hiererchy"]
                        self.train_dls[0] = torch.utils.data.DataLoader(
                            clustered_sampler.ds, batch_size=int(os.environ["batch_size"]), sampler=clustered_sampler,
                            num_workers=0,pin_memory=True)
                        self.models[0].clustering_algorithm = None
                    else:
                        print("loading before start clustering")
                        pkl_filename = f"ckpt/{self.tb.exp_name}_cluster_model.pkl"
                        with open(pkl_filename, 'rb') as file:
                            self.models[0].clustering_algorithm.model = pickle.load(file)
                        pkl_filename = f"ckpt/{self.tb.exp_name}_resnet_cluster_dict.pkl"
                        with open(pkl_filename, 'rb') as file:
                            self.models[0].cluster_dict = pickle.load(file)

            print(f"running from steps {self.curr_steps}")
        self.start_clustering = start_clustering
        self.clustered_sampler = clustered_sampler

        for phase in ["train", "eval", "test"]:
            self.times[phase] = [[] for _ in models]
            self.losses[phase] = [[] for _ in models]
            self.accuracies[phase] = [[] for _ in models]
            self.steps_for_acc_loss_and_time[phase] = [[] for _ in models]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_ckpt(self, model, optimizer, step, idx, exp_name):

        if step - self.last_save[idx] > 1000:
            self.last_save[idx] = step
            print(f"start saving model for idx {idx}")
            state_to_save = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "sched_state_cidt": self.schedulers[idx].state_dict()
            }
            if idx == 0:
                if model.clustering_algorithm is not None:
                    pkl_filename = f"ckpt/{exp_name}_cluster_model.pkl"
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(model.clustering_algorithm.model, file)
                    pkl_filename = f"ckpt/{exp_name}_resnet_cluster_dict.pkl"
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(model.cluster_dict, file)
                else:
                    pkl_filename = f"ckpt/{exp_name}_sampler_cluster_dict.pkl"
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(self.train_dls[idx].sampler.cluster_dict, file)
                    state_to_save["center"] = self.train_dls[idx].sampler.center
                    state_to_save["hiererchy"] = self.train_dls[idx].sampler.hiererchy
            torch.save(state_to_save, f"ckpt/model_{exp_name}_{idx}.pth")
            print(f"model saved step is: {step} idx: {idx}")

    def run_train(self, idx):

        model = self.models[idx]
        curr_step = self.curr_steps[idx]

        dl = self.train_dls[idx]
        running_loss = 0.0
        running_corrects = 0

        optimizer = self.optimizers[idx]
        num_examples = 0

        model.train()
        since = time.time()
        epoch_viz = False
        bar = tqdm(dl)
        for (inputs, images_indexes), labels in bar:
            if not epoch_viz:
                # temp_labels = [label_to_str[x.item()] for x in labels[:20]]
                self.tb.add_images(idx=idx, images=inputs[:20], title=f"train", step=curr_step)
                epoch_viz = True
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

            outputs = model(inputs, images_indexes)

            loss = self.loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            curr_step += 1
            bar.set_description(f"idx: {idx} step {curr_step} loss {loss:.3f}")
            loss.backward()
            optimizer.step()
            self.schedulers[idx].step()
            running_loss += loss.cpu().item() * inputs.size(0)
            num_examples += inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).cpu()
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / num_examples
        epoch_acc = running_corrects.double() / num_examples

        if self.last_bar_update < curr_step and idx == 0:
            # self.bar.update(curr_step)
            self.last_bar_update = curr_step
        return time_elapsed, epoch_loss, epoch_acc, 0,self.schedulers[idx].get_last_lr()

    def run_eval(self, idx):
        model = self.models[idx]
        curr_step = self.curr_steps[idx]
        model.eval()
        dl = self.eval_dls[idx]
        if curr_step < self.start_clustering:
            return 0, 0, 0, 0,0
        running_loss = 0.0
        running_corrects = 0
        since = time.time()
        eval_loss_dict = {}
        optimizer = self.optimizers[idx]
        num_examples = 0
        for (inputs, images_indexes), labels in dl:
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
                outputs = model(inputs, images_indexes)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if model.do_clustering():
                    losses = self.loss_fn_eval(outputs, labels)
                    for curr_input, images_index, temp_loss in zip(inputs, images_indexes, losses):
                        eval_loss_dict[int(images_index)] = temp_loss
                running_loss += loss.cpu().item() * inputs.size(0)
                num_examples += inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu()
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / num_examples
        epoch_acc = running_corrects.double() / num_examples
        if model.do_clustering() and self.start_clustering <= curr_step and self.clusters is None:
            self.clusters = model.get_clusters()
            self.train_dls[idx] = torch.utils.data.DataLoader(
                self.clustered_sampler.ds, batch_size=int(os.environ["batch_size"]), sampler=self.clustered_sampler,
                num_workers=0,pin_memory=True)
            self.train_dls[idx].sampler.create_distribiouns(self.clusters, eval_loss_dict)

            model.clustering_algorithm = None

        return time_elapsed, epoch_loss, epoch_acc, 0, 0

    def run_test(self, idx):
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
        num_examples = 0
        model.test_time_activate()
        epoch_viz = False
        for (inputs, image_indexes), labels in dl:
            if inputs.shape[0] == 1:
                print("skipped")
                continue
            if not epoch_viz:
                # temp_labels = [label_to_str[x.item()] for x in labels[:20]]
                self.tb.add_images(idx=idx, images=inputs[:20], title=f"test ", step=curr_step)
                epoch_viz = True

            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()

            with torch.no_grad():
                optimizer.zero_grad()
                model.zero_grad()
                outputs = model(inputs, image_indexes)

                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if self.clusters and idx == 0:
                    is_in_train = self.train_dls[idx].sampler.is_in_train(image_indexes)
                    assert len(is_in_train) == len(preds)
                    for pred, flag, label in zip(preds, is_in_train, labels.data):
                        if flag:
                            if pred == label:
                                sub_running_corrects += 1
                            sub_running_corrects_disc += 1
                running_loss += loss.cpu().item() * inputs.size(0)
                num_examples += inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu()
        model.done_test()
        self.curr_steps[idx] = curr_step

        time_elapsed = time.time() - since
        epoch_loss = running_loss / num_examples
        epoch_acc = running_corrects.double() / num_examples
        try:
            self.schedulers[idx].update_sched(epoch_acc)
        except AttributeError:
            pass
        if sub_running_corrects_disc == 0:
            sub_epoch_acc = None
        else:
            sub_epoch_acc = sub_running_corrects / sub_running_corrects_disc

        return time_elapsed, epoch_loss, epoch_acc, sub_epoch_acc,0

    def save_epoch_results(self, phase, idx, curr_time, loss, acc, sub_acc,lr, step):

        def save_results_to_tb():
            self.tb.add_scalar(idx, phase + " loss", loss, step)
            self.tb.add_scalar(idx, phase + " curr_time", curr_time, step)
            self.tb.add_scalar(idx, phase + " acc", acc, step)
            if lr:
                self.tb.add_scalar(idx, phase + " lr", lr[0], step)
            if sub_acc:
                self.tb.add_scalar(idx, phase + " sub_acc", sub_acc, step)

        def print_results():
            print(f"idx: {idx}, phase: {phase}, loss: {loss:.3f}, acc: {acc:.3f}\n")

        self.times[phase][idx].append(curr_time)
        self.losses[phase][idx].append(loss)
        self.accuracies[phase][idx].append(acc.item())
        self.steps_for_acc_loss_and_time[phase][idx].append(self.curr_steps[idx])
        save_results_to_tb()
        print_results()

    def train_models(self):
        print("start trining")
        while True:
            if min(self.curr_steps) > self.num_steps:
                break
            idx = np.argmin(self.curr_steps)
            assert idx in [0, 1]  # todo: remove

            curr_time, loss, acc, sub_acc,curr_lr = self.run_train(idx)
            self.train_dls[idx].dataset.collect_garbage()
            if (curr_time, loss, acc, sub_acc) != (0, 0, 0, 0):
                self.save_epoch_results("train", idx, curr_time, loss, acc, sub_acc,lr=curr_lr,step=self.curr_steps[idx])
            if self.models[idx].do_clustering():

                curr_time, loss, acc, sub_acc,_ = self.run_eval(idx)
                self.eval_dls[idx].dataset.collect_garbage()
                if (curr_time, loss, acc, sub_acc) != (0, 0, 0, 0):
                    self.save_epoch_results("eval", idx, curr_time, loss, acc, sub_acc, step = self.curr_steps[idx],lr=None)

            curr_time, loss, acc, sub_acc,_ = self.run_test(idx)
            self.test_dls[idx].dataset.collect_garbage()
            if (curr_time, loss, acc, sub_acc) != (0, 0, 0, 0):
                self.save_epoch_results("test", idx, curr_time, loss, acc, sub_acc, step = self.curr_steps[idx],lr=None)

            self.save_ckpt(model=self.models[idx], optimizer=self.optimizers[idx], step=self.curr_steps[idx], idx=idx,
                           exp_name=self.tb.exp_name)
