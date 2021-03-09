import torch
class chainedCyclicLr(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,optimizer,base_lrs,max_lrs,step_sizes_up,ths):
        self.i = 0
        self.base_lrs = base_lrs
        self.max_lrs = max_lrs
        self.step_sizes_up = step_sizes_up
        self.optimizer = optimizer
        self.ths = ths
        self.current = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lrs[0], max_lr=max_lrs[0], step_size_up=step_sizes_up[0],
                                          mode="triangular2")
    def update_sched(self,val_acc):
        if val_acc > self.ths[self.i]:
            print(f"switch to sched {self.i} with params {self.base_lrs[self.i]},")
            self.i+=1
            self.current = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.base_lrs[self.i], max_lr=self.max_lrs[self.i],
                                                             step_size_up=self.step_sizes_up[self.i],
                                                             mode="triangular2")


    def get_lr(self):
        return self.current.get_lr()
    def get_last_lr(self):
        return self.current.get_last_lr()
    def step(self):
        return self.current.step()
    def state_dict(self):
        state_dict =self.current.state_dict()
        state_dict["base_lrs"] = self.base_lrs
        state_dict["max_lrs"] = self.max_lrs
        state_dict["step_sizes_up"] = self.step_sizes_up
        state_dict["ths"] = self.ths
        state_dict["i"] = self.i
        return state_dict
    def load_state_dict(self,state_dict):
        self.base_lrs = state_dict.pop("base_lrs")
        self.i = state_dict.pop("i")
        self.max_lrs = state_dict.pop("max_lrs")
        self.step_sizes_up = state_dict.pop("step_sizes_up")
        self.ths = state_dict.pop("ths")
        self.current = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.base_lrs[self.i],
                                                         max_lr=self.max_lrs[self.i],
                                                         step_size_up=self.step_sizes_up[self.i],
                                                         mode="triangular2")
        self.current.load_state_dict(state_dict)
