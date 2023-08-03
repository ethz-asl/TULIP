import torch
import math


class MAEEvaluator(object):
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, output):
        y_pred, y = output[0].detach().flatten(), output[1].detach().flatten()
        abs_error = torch.abs(y_pred - y)

        self.sum += torch.sum(abs_error)
        self.cnt += abs_error.shape[0]         # Number of samples

    def compute(self):
        return float(self.sum / self.cnt) if self.cnt > 0 else math.inf
