import torch.optim as optim
import math
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, mode, factor, patience, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._makeOptimizer()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True)


    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm
    def lronplateau(self, loss):
        self.scheduler.step(loss)
    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if epoch == 16:
            self.lr /= 100
            print(f"[第七轮强制调整] 新学习率: {self.lr:.2e}")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr


        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = False
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = False

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        # self._makeOptimizer()
