
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/metrics.ipynb

import torch
from torch import tensor

from exp.nb_utils import snakify_class_name

class Metric:
    def reset(self): pass
    def accumulate(self, learner): pass

    @property
    def value(self): raise NotImplementedError

    @property
    def name(self): return snakify_class_name(self, "Metric")

class AvgMetric(Metric):
    def __init__(self, func):
        self.func = func

    def reset(self):
        self.total = tensor(0.)
        self.count = 0

    def accumulate(self, learner):
        bs = learner.xb.shape[0]
        self.total += self.func(learner.pred, learner.yb).detach().cpu() * bs
        self.count += bs

    @property
    def value(self):
        return self.total / self.count if self.count else None

    @property
    def name(self):
        return self.func.__name__

class AvgLoss(Metric):
    def reset(self):
        self.total = tensor(0.)
        self.count = 0

    def accumulate(self, learner):
        bs = learner.xb.shape[0]
        self.total += learner.loss.detach().cpu().mean() * bs
        self.count += bs

    @property
    def value(self):
        return self.total / self.count if getattr(self, 'count') else None

    @property
    def name(self): return "loss"

class AvgSmoothLoss(Metric):
    def __init__(self, beta=0.98):
        self.beta = beta

    def reset(self):
        self.val = tensor(0.)
        self.count = 0

    def accumulate(self, learner):
        self.count += 1
        self.val = torch.lerp(learner.loss.detach().cpu().mean(),
                              self.val,
                              self.beta)

    @property
    def value(self):
        return self.val/(1-self.beta**self.count)

    @property
    def name(self): return "smooth_loss"