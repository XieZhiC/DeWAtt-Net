import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable


def mape_loss(target, input):
    return (torch.abs(input - target) / (torch.abs(target) + 1e-2)).mean() * 100


def MAPE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))

    # e = (y_true + y_pre) / 2 + 1e-2
    # re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    re = np.mean(np.abs((y_true - y_pre) / y_true)) * 100

    return re


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',', skiprows=1)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self.train_feas = self.dat[:int(train * self.n), :]
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale[:3]).float()

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.device = device

    def _normalized(self, normalize):

        for i in range(self.m):
            self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
            self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))


    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        # print("datshape", self.dat.shape)
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P

            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i] + 1 - horizon:idx_set[i] + 1, :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size



