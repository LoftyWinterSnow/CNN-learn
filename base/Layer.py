import numpy as np
from typing import Callable, List, Tuple, Union, Optional, TYPE_CHECKING, overload
if TYPE_CHECKING:
    from .Graph import Graph
from .c_functions import CFunc as CF
from .Variable import Variable


class Layer:

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.child: List[Variable] = []
        self.parent: List[Variable] = []
        self.wait_backward = False
        self.initialized = False

    def init_var(
        self,
        graph: 'Graph',
        input_var: Union[List[Variable], Variable],
        output_var: Union[List[Variable], Variable],
    ):
        if not self.initialized:
            self.input_var = input_var
            self.output_var = output_var
            self.graph = graph
            self.name = self.layer_name + str(self.graph.layerNum)
            self.graph.register(input_var, output_var, self)

        self.initialized = True

    def _forward(self, *args, **kargs) -> Union[np.ndarray, Tuple[np.ndarray]]:
        pass

    def _backward(self, *args,
                  **kargs) -> Union[np.ndarray, Tuple[np.ndarray]]:
        pass

    def backward(self, no_grad=False):
        if self.wait_backward:
            for child in self.child:
                self.graph.scope[child].grad_eval(no_grad=no_grad)
            grad = self._backward(*self.get_output_grad())
            if isinstance(grad, tuple):
                self.set_input_grad(*grad)
            else:
                self.set_input_grad(grad)
            if not no_grad:
                self.wait_backward = False
        else:
            pass

    def forward(self, no_grad=False):
        if self.wait_backward:
            pass
        else:
            for parent in self.parent:
                self.graph.scope[parent].eval(no_grad=no_grad)
            res = self._forward(*self.get_input())
            if isinstance(res, tuple):
                self.set_output(*res)
            else:
                self.set_output(res)
            self.clear_input_grad()
            if not no_grad:
                self.wait_backward = True

    def __call__(self, *args, **kargs):
        res = self._forward(*args, *kargs)
        return res

    def clear_input_grad(self):
        if isinstance(self.input_var, list):
            for var in self.input_var:
                if var.requires_grad:
                    var.grad.data *= 0
        else:
            self.input_var.grad.data *= 0

    def set_output(self, *args: np.ndarray):
        if isinstance(self.output_var, list):
            for i, var in enumerate(self.output_var):
                var.set_data(args[i])
        else:
            self.output_var.set_data(args[0])

    def set_input_grad(self, *args: np.ndarray):
        if isinstance(self.input_var, list):
            for i, var in enumerate(self.input_var):
                if var.requires_grad:
                    var.grad.add_data(args[i])
        else:
            self.input_var.grad.add_data(args[0])

    def get_input(self):
        if isinstance(self.input_var, list):
            return [var.data for var in self.input_var]
        else:
            return [self.input_var.data]

    def get_output_grad(self):
        if isinstance(self.output_var, list):
            return [var.grad.data for var in self.output_var]
        else:
            return [self.output_var.grad.data]


class Flatten(Layer):

    def __init__(self, input_var: Optional[Variable] = None):
        Layer.__init__(self, 'Flatten')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def _forward(self, x: np.ndarray):
        return x.reshape(x.shape[0], -1)

    def _backward(self, grad: np.ndarray):
        return grad.reshape(self.input_var.shape)

    def __repr__(self):
        return f'Flatten()'


class Linear(Layer):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 input_var: Optional[Variable] = None):
        Layer.__init__(self, 'Linear')
        self.in_features = in_features
        self.out_features = out_features

        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def init_var(self, graph, input_var, output_var):
        super().init_var(graph, input_var, output_var)
        self.weight = Variable(
            np.random.randn(self.in_features, self.out_features) /
            np.sqrt(self.in_features / 2),
            learnable=True,
            graph=self.graph)
        self.bias = Variable(np.random.randn(self.out_features) /
                             np.sqrt(self.in_features / 2),
                             learnable=True,
                             graph=self.graph)

    def _forward(self, x: np.ndarray):
        return x @ self.weight.data + self.bias.data

    def _backward(self, grad: np.ndarray):
        for i in range(grad.shape[0]):
            # x[i][:, np.newaxis] 将一维向量转置 数学上等同于x.T 形式上等同于 x.reshape((-1, 1))
            self.weight.grad.data += self.input_var.data[
                i][:, np.newaxis] @ grad[i][np.newaxis, :]
            # x.T @ grad
            self.bias.grad.data += grad[i]
        new_grad = grad @ self.weight.data.T
        return new_grad

    def __repr__(self):
        return 'Linear(in_features=%d, out_features=%d)' % (self.in_features,
                                                            self.out_features)


class CrossEntropyLoss(Layer):

    def __init__(self,
                 input_var: Optional[Variable] = None,
                 label: Optional[Variable] = None):
        Layer.__init__(self, 'CrossEntropyLoss')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, [input_var, label],
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return "CrossEntropyLoss()"

    def _forward(self, x: np.ndarray, label: np.ndarray):
        """
        交叉熵损失\n
        首先对x进行softmax\n
        ``pred = exp(x)/sum(exp(x))`` \n
        随后进行交叉熵损失计算 \n
        ``loss = -sum(lab * log(pred))`` \n
        最后对于多个batch进行累加\n
        式中, label应为one-hot编码, 但是事实上输入的label并非one-hot编码 \n
        所以需要将lab变量抽离, 由于label表示实际的概率分布, 故只有真实值对应的值为1, 其余均为0, 故可以将该式简化为 \n
        ``loss = -log(pred_t)`` \n
        其中t为label中为1的索引, 可以直接使用默认label输入 \n
        继续展开得到 \n
        ``loss = -log(exp(x_t)/sum(exp(x))) = -x_t + log(sum(exp(x)))`` \n   
        """
        loss = 0
        for i in range(x.shape[0]):
            loss += -x[i, label[i]] + np.log(np.sum(np.exp(x[i])))
        return loss

    def _backward(self, *args):
        grad = np.array(
            list(
                map(lambda vec: np.exp(vec) / np.sum(np.exp(vec)),
                    self.input_var[0].data)))
        for i in range(grad.shape[0]):
            grad[i, self.input_var[1].data[i]] -= 1
        return grad


class ReLU(Layer):

    def __init__(self, input_var: Optional[Variable] = None):
        Layer.__init__(self, 'ReLU')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return 'ReLU()'

    def _forward(self, x: np.ndarray):
        return np.maximum(x, 0)

    def _backward(self, grad: np.ndarray):
        new_grad = grad
        new_grad[self.input_var.data < 0] = 0
        return new_grad


class LeakyReLU(Layer):

    def __init__(self, k, input_var: Optional[Variable] = None):
        self.k = k
        Layer.__init__(self, 'LeakyReLU')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return 'LeakyReLU()'

    def _forward(self, x: np.ndarray):
        return np.maximum(x, self.k * x)

    def _backward(self, grad: np.ndarray):
        new_grad = grad
        new_grad[self.input_var.data < 0] = self.k
        return new_grad


class ELU(Layer):

    def __init__(self, alpha, input_var: Optional[Variable] = None):
        self.alpha = alpha
        Layer.__init__(self, 'ELU')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return 'ELU()'

    def _forward(self, x: np.ndarray):
        indx = x < 0
        x[indx] = self.alpha * (np.exp(x[indx]) - 1)
        return x

    def _backward(self, grad: np.ndarray):
        new_grad = grad
        indx = self.input_var.data < 0
        new_grad[indx] = self.alpha * np.exp(self.input_var.data[indx])
        return new_grad


class Sigmoid(Layer):

    def __init__(self, input_var: Optional[Variable] = None):
        Layer.__init__(self, 'Sigmoid')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return 'Sigmoid()'

    def _forward(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _backward(self, grad: np.ndarray):
        new_grad = grad * self._forward(
            self.input_var.data) * (1 - self._forward(self.input_var.data))
        return new_grad



class Tanh(Layer):

    def __init__(self, input_var: Optional[Variable] = None):
        Layer.__init__(self, 'Tanh')
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return 'Tanh()'

    def _forward(self, x: np.ndarray):
        return np.tanh(x)

    def _backward(self, grad: np.ndarray):
        new_grad = grad * (1 - np.tanh(self.input_var.data)**2)
        return new_grad


class Sequence(Layer):

    def __init__(self,
                 layers: List[Layer],
                 input_var: Optional[Variable] = None):
        self.wait_backward = False
        self.initialized = False
        self.layers = layers
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    @property
    def parent(self):
        return self.layers[0].parent

    @property
    def output_var(self):
        return self.layers[-1].output_var

    @property
    def input_var(self):
        return self.layers[0].input_var

    @property
    def child(self):
        return self.layers[-1].child

    def __getitem__(self, index):
        return self.layers[index]

    def __repr__(self):
        return 'Sequence(\n\t' + ',\n\t'.join(map(str, self.layers)) + ')'

    def init_var(
        self,
        graph: 'Graph',
        input_var: Union[List[Variable], Variable],
        output_var: Union[List[Variable], Variable],
    ):
        if not self.initialized:
            self.graph = graph
            self.layers[0].init_var(graph, input_var,
                                    Variable(graph=input_var.graph))
            for i in range(1, len(self.layers) - 1):
                self.layers[i].init_var(graph, self.layers[i - 1].output_var,
                                        Variable(graph=input_var.graph))
            self.layers[-1].init_var(graph, self.layers[-2].output_var,
                                     output_var)

        self.initialized = True

    def _forward(self, x: np.ndarray):
        res = x
        for layer in self.layers:
            res = layer._forward(res)
        return res

    def forward(self, no_grad=False):
        if self.wait_backward:
            pass
        else:
            self.output_var.eval(no_grad=no_grad)
            if not no_grad:
                self.wait_backward = True

    def backward(self, no_grad=False):
        if self.wait_backward:
            self.input_var.grad_eval(no_grad=no_grad)
            if not no_grad:
                self.wait_backward = False
        else:
            pass


class Conv2d(Layer):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 input_var: Optional[Variable] = None):
        Layer.__init__(self, 'Conv2d')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def init_var(self, graph, input_var, output_var):
        super().init_var(graph, input_var, output_var)
        self.weight = Variable(
            np.random.randn(self.out_channels, self.in_channels,
                            self.kernel_size, self.kernel_size) /
            np.sqrt(
                self.kernel_size * self.kernel_size * self.in_channels / 2),
            learnable=True,
            graph=graph)
        self.bias = Variable(np.random.randn(self.out_channels) / np.sqrt(
            self.kernel_size * self.kernel_size * self.in_channels / 2),
                             learnable=True,
                             graph=graph)

    def __repr__(self):
        return f'Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def _forward(self, x: np.ndarray):
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                       (self.padding, self.padding)),
                   'constant',
                   constant_values=0)
        out, self.col_img = CF.conv2d(x, self.weight.data, self.bias.data,
                                      self.stride)
        return out

    def _backward(self, grad: np.ndarray):
        col_grad = grad.reshape((grad.shape[0], self.out_channels, -1))
        for i in range(grad.shape[0]):
            self.weight.grad.data += (
                self.col_img[i].T @ col_grad[i].T).reshape(self.weight.shape)
        self.bias.grad.data += np.sum(col_grad, axis=(2, 0))
        #TODO: 增加对于stride>1情况的支持
        pad_grad = np.pad(grad, ((0, 0), (0, 0), (self.padding, self.padding),
                                 (self.padding, self.padding)),
                          'constant',
                          constant_values=0)
        next_grad = CF.deconv2d(pad_grad, self.input_var.shape,
                                self.weight.data, self.kernel_size,
                                self.stride)
        return next_grad


class MaxPool2d(Layer):

    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2,
                 input_var: Optional[Variable] = None):
        Layer.__init__(self, 'MaxPool2d')
        self.kernel_size = kernel_size
        self.stride = stride
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var,
                          Variable(graph=input_var.graph))

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"

    def _forward(self, x: np.ndarray):
        out, self.mask = CF.maxpool2d(x, self.kernel_size, self.stride)
        return out

    def _backward(self, grad: np.ndarray):
        return np.repeat(np.repeat(grad, self.stride, axis=2),
                         self.stride,
                         axis=3) * self.mask


class Add(Layer):

    def __init__(self,
                 input_var_1: Optional[Variable] = None,
                 input_var_2: Optional[Variable] = None):
        Layer.__init__(self, 'Add')
        if not isinstance(input_var_1, type(None)) and not isinstance(
                input_var_2, type(None)):
            self.init_var(input_var_1.graph, [input_var_1, input_var_2],
                          Variable(graph=input_var_1.graph))

    def _forward(self, x1: np.ndarray, x2: np.ndarray):
        return x1 + x2

    def _backward(self, grad: np.ndarray):
        return grad, grad


if __name__ == '__main__':
    x = Variable(np.random.randint(0, 5, (5, 1, 28, 28)))
    x1 = Sequence([
        Flatten(),
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ], x)
    y = CrossEntropyLoss(x1.output_var,
                         Variable(np.random.randint(0, 10, 5), dtype=np.uint8))

    # f1.init_var(x.graph, x, Variable(graph=x.graph))
    import code
    code.interact(local=locals())
