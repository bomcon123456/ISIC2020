
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/schedules.ipynb

from functools import wraps
import math

import torch
from torch import tensor

class _Annealer:
    def __init__(self, f, start, end):
        self.f, self.start, self.end = f, start, end

    def __call__(self, pos):
        return self.f(self.start, self.end, pos)

def annealer(f):
    @wraps(f)
    def _inner(start, end):
        return _Annealer(f, start, end)
    return _inner

def sched_lin(start, end, pos):
    return start + pos*(end-start)


def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2


def sched_no(start, end, pos):
    return start


def sched_exp(start, end, pos):
    return start * (end/start) ** pos


def SchedLin(start, end):
    return _Annealer(sched_lin, start, end)


def SchedCos(start, end):
    return _Annealer(sched_cos, start, end)


def SchedNo(start, end):
    return _Annealer(sched_no,  start, end)


def SchedExp(start, end):
    return _Annealer(sched_exp, start, end)


def SchedPoly(start, end, power):
    def _inner(pos): return start + (end - start) * pos ** power
    return _inner

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + pcts)
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        if pos == 1.:
            return scheds[-1](1.)
        idx = (pos >= pcts).nonzero(as_tuple=False).max()
        if idx == len(pcts) - 1:
            return scheds[-1](pos)
        actual_pos = (pos - pcts[idx]) / (pcts[idx+1] - pcts[idx])
        return scheds[idx](actual_pos.item())
    return _inner

def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`→`middle` & `middle`→`end`"
    return combine_scheds([pct,1-pct], [SchedCos(start, middle), SchedCos(middle, end)])