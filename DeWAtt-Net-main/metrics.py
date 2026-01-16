import math

import numpy as np


def MAE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = np.abs(y_true - y_pre).mean()
    return re

def torch_MAE(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return torch.mean(torch.abs(y_true - y_pred))

def RMSE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = math.sqrt(((y_true - y_pre) ** 2).mean())
    return re


def MAPE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))

    # e = (y_true + y_pre) / 2 + 1e-2
    # re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    re = np.mean(np.abs((y_true - y_pre) / y_true)) * 100

    return re

def MSE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = ((y_true - y_pre) ** 2).mean()
    return re