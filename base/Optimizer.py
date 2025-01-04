import numpy as np
from typing import List
from .Graph import Graph
from .Variable import Variable


class Optimizer:

    def __init__(self, graph: Graph, weight_decay=1e-6):
        self.params: List[Variable] = []
        self.weight_decay = weight_decay
        for value in graph.scope.values():
            if isinstance(value, Variable):
                if value.learnable:
                    self.params.append(value)

    def gradient_descent(self, batch_size):
        ...


class SGD(Optimizer):

    def __init__(self, graph: Graph, weight_decay=1e-6):
        Optimizer.__init__(self, graph, weight_decay)

    def gradient_descent(self, batch_size, lr=1e-4):
        for var in self.params:
            var.data *= (1 - self.weight_decay)
            var.data -= lr * var.grad.data / batch_size
            var.grad.data *= 0


class Adam(Optimizer):

    def __init__(self,
                 graph: Graph,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 weight_decay=1e-6):
        Optimizer.__init__(self, graph, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.var_t = np.zeros_like(self.params)  # 记录每个参数的更新次数
        self.m_t = [np.zeros_like(var.grad.data)
                    for var in self.params]  # 一阶矩估计
        self.v_t = [np.zeros_like(var.grad.data)
                    for var in self.params]  # 二阶矩估计

    def gradient_descent(self, batch_size, lr=1e-4):
        for i, var in enumerate(self.params):
            var.data *= (1 - self.weight_decay)
            self.var_t[i] += 1
            lr_t = lr * np.sqrt(1 - pow(self.beta2, self.var_t[i])) / (
                1 - pow(self.beta1, self.var_t[i]))
            self.m_t[i] = self.beta1 * self.m_t[i] + (
                1 - self.beta1) * var.grad.data / batch_size
            self.v_t[i] = self.beta2 * self.v_t[i] + (1 - self.beta2) * (
                (var.grad.data / batch_size)**2)
            var.data -= lr_t * self.m_t[i] / (self.v_t[i] + self.eps)**0.5
            var.grad.data *= 0
