import json
import os
import subprocess
import time

import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from utils import get_md5sum, Tb
import numpy as np


class Trainer:
    def __init__(self, models, train_dls, eval_dls, test_dls, loss_fn, loss_fn_eval, optimizers, num_steps, tb):
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

        for phase in ["train", "eval", "test"]:
            self.times[phase] = [[] for _ in models]
            self.losses[phase] = [[] for _ in models]
            self.accuracies[phase] = [[] for _ in models]
            self.steps_for_acc_loss_and_time[phase] = [[] for _ in models]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run_epoch(self, idx, phase):
        model = self.models[idx]
        if phase == "train":
            dl = self.train_dls[idx]
        elif phase == "eval":
            dl = self.eval_dls[idx]
            if not dl:
                return 0, 0, 0
        else:
            assert phase == "test"
            dl = self.test_dls[idx]
        running_loss = 0.0
        running_corrects = 0

        sub_running_corrects = 0
        sub_running_corrects_disc = 0
        since = time.time()
        eval_loss_dict = {}
        curr_step = self.curr_steps[idx]
        optimizer = self.optimizers[idx]

        for inputs, labels in dl:
            if inputs.shape[0] == 1:
                print("skipped")
                continue
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == "train":
                    curr_step += 1
                    if curr_step % 1000 == 0:

                        print(f"curr step is {curr_step} in model {idx}")
                    loss.backward()
                    optimizer.step()
                elif phase == "eval" and model.do_clustering():
                    losses = self.loss_fn_eval(outputs, labels)
                    for curr_input, temp_loss in zip(inputs, losses):
                        hashed = get_md5sum(curr_input.cpu().numpy().tobytes())
                        eval_loss_dict[str(hashed)] = temp_loss
                running_loss += loss.item() * inputs.size(0)
                if phase == "test" and model.do_clustering():
                    cluster_labels = model.clustering_algorithm.predict(inputs,model.cluster_dict)
                    is_in_train = self.train_dls[idx].sampler.is_in_train(cluster_labels)
                    assert len(is_in_train) == len(preds)
                    for pred, flag, label in zip(preds, is_in_train, labels.data):
                        if flag:
                            if pred == label:
                                sub_running_corrects += 1
                            sub_running_corrects_disc += 1
                running_corrects += torch.sum(preds == labels.data)
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / len(dl.dataset)
        epoch_acc = running_corrects.double() / len(dl.dataset)
        if sub_running_corrects_disc == 0:
            sub_epoch_acc = None
        else:
            sub_epoch_acc = sub_running_corrects / sub_running_corrects_disc
        if phase == "eval" and model.do_clustering() and self.train_dls[idx].sampler.start_clustering < curr_step:
            if self.clusters is None:
                self.clusters = model.get_clusters()
            ret_value = self.train_dls[idx].sampler.create_distribiouns(self.clusters, eval_loss_dict, curr_step)  # todo: run this line only once
            if ret_value == "done":
                model.clustering_algorithm = None
                self.train_dls[idx] = torch.utils.data.DataLoader(
                    self.train_dls[idx].dataset + self.eval_dls[idx].dataset, batch_size=32, shuffle=True,
                    num_workers=0 if os.environ["my_computer"] == "True" else 2)
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
            for phase in ["train", "eval", "test"]:
                if phase == "eval" and not self.models[idx].do_clustering():
                    continue
                curr_time, loss, acc, sub_acc = self.run_epoch(idx, phase)
                self.save_epoch_results(phase, idx, curr_time, loss, acc, sub_acc, self.curr_steps[idx])
