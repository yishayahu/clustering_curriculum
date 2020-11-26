import json
import subprocess
import time

import torch
import torchvision.transforms.functional as F
from utils import get_md5sum,Tb
import numpy as np


class Trainer:
    def __init__(self, models, train_dls, eval_dls, test_dls, loss_fn, loss_fn_eval, optimizers, num_steps,tb):
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
                return 0,0,0
        else:
            assert phase == "test"
            dl = self.test_dls[idx]
        running_loss = 0.0
        running_corrects = 0
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
                    loss.backward()
                    optimizer.step()
                elif phase == "eval" and model.do_clustering():
                    losses = self.loss_fn_eval(outputs, labels)
                    for curr_input, temp_loss in zip(inputs, losses):
                        hashed = get_md5sum(curr_input.cpu().numpy().tobytes())
                        eval_loss_dict[str(hashed)] = temp_loss
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        self.curr_steps[idx] = curr_step
        time_elapsed = time.time() - since
        epoch_loss = running_loss / len(dl.dataset)
        epoch_acc = running_corrects.double() / len(dl.dataset)
        if phase == "eval" and model.do_clustering():
            self.train_dls[idx].sampler.create_distribiouns(model.get_clusters(), eval_loss_dict, curr_step)
        return time_elapsed, epoch_loss, epoch_acc
    def save_epoch_results(self,phase,idx,curr_time,loss,acc,step):
        def save_results_to_json():
            json.dump(self.times, open("times.json", 'w'))
            json.dump(self.losses, open("losses.json", 'w'))
            json.dump(self.accuracies, open("accuracies.json", 'w'))
            json.dump(self.steps_for_acc_loss_and_time, open("steps.json", 'w'))
        def save_results_to_tb():
            self.tb.add_scalar(idx,phase + " loss",loss,step)
            self.tb.add_scalar(idx,phase + " curr_time",curr_time,step)
            self.tb.add_scalar(idx,phase + " acc",acc,step)
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
            assert idx in [0,1] # todo: remove
            for phase in ["train", "eval", "test"]:
                if phase == "eval" and not self.models[idx].do_clustering():
                    continue
                curr_time, loss, acc = self.run_epoch(idx, phase)
                self.save_epoch_results(phase,idx,curr_time,loss,acc,self.curr_steps[idx])


