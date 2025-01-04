import numpy as np
import struct
from tqdm import tqdm
from typing import *
import matplotlib.pyplot as plt
from base.c_functions import CFunc as F
# from NN_base import *

data = "D:/python project/venv/"

train_img_p = 'train-images.idx3-ubyte'
train_lab_p = 'train-labels.idx1-ubyte'

test_img_p = 't10k-images.idx3-ubyte'
test_lab_p = 't10k-labels.idx1-ubyte'


def load_img(file):
    with open(file, 'rb') as f:
        struct.unpack('>4i', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(-1, 1, 28, 28)
    return img / 256


def load_lab(file):
    with open(file, 'rb') as f:
        struct.unpack('>2i', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8).reshape(-1)
    return lab


train_img = load_img(data + train_img_p)
train_lab = load_lab(data + train_lab_p)
test_img = load_img(data + test_img_p)
test_lab = load_lab(data + test_lab_p)


class Layer:

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, *args, **kwds):
        pass

    def gradient(self, *args, **kwds):
        pass

    def backward(self, *args, **kwds):
        pass


class Sequence(Layer):

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        new_grad = grad
        for layer in reversed(self.layers):
            new_grad = layer.backward(new_grad)
        return new_grad

    def gradient(self, learn_rate=1e-4):
        for layer in self.layers:
            layer.gradient(learn_rate=learn_rate)


class Conv2d(Layer):
    # TODO: 使用C++加速
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0):
        """
        卷积层
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小(最好为奇数)
            stride: 步长 默认为1
            padding: 填充 默认为0
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = np.random.randn(
            out_channels, self.in_channels, kernel_size,
            kernel_size) / np.sqrt(
                kernel_size * kernel_size * self.in_channels / 2)
        self.bias = np.random.randn(out_channels) / np.sqrt(
            kernel_size * kernel_size * self.in_channels / 2)

        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, x):  # 前向传播
        """
        Args:
            x: 输入数据 为[batch, channel, H, W]的格式
        """

        self.input_shape = x.shape
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                       (self.padding, self.padding)),
                   'constant',
                   constant_values=0)
        # N, C, H, W = x.shape
        # col_weight = self.weight.reshape((self.out_channels, -1)).T
        # self.col_img = []
        # out = np.empty(
        #     (N, self.out_channels, (H - self.kernel_size) // self.stride + 1,
        #      (W - self.kernel_size) // self.stride + 1))
        # for i in range(N):
        #     img_i = x[i][np.newaxis, :]
        #     col_img_i = self.im2col(img_i)
        #     out[i] = (col_img_i @ col_weight + self.bias).T.reshape(
        #         out.shape[1:])
        #     self.col_img.append(col_img_i)
        # self.col_img = np.array(self.col_img)
        out, self.col_img = F.conv2d(x, self.weight, self.bias, self.stride)
        return out

    def backward(self, grad):
        col_grad = grad.reshape((grad.shape[0], self.out_channels, -1))
        for i in range(grad.shape[0]):
            self.weight_grad += (self.col_img[i].T @ col_grad[i].T).reshape(
                self.weight.shape)
        self.bias_grad += np.sum(col_grad, axis=(2, 0))
        #TODO: 增加对于stride>1情况的支持
        pad_grad = np.pad(grad, ((0, 0), (0, 0), (self.padding, self.padding),
                                 (self.padding, self.padding)),
                          'constant',
                          constant_values=0)
        # flip_weight = np.flip(self.weight, axis=(2, 3))
        # col_flip_weight = flip_weight.reshape((self.in_channels, -1)).T
        # next_grad = np.array(
        #     list(
        #         map(
        #             lambda img:
        #             (self.im2col(img[np.newaxis, :]) @ col_flip_weight
        #              ).T.reshape(self.input_shape[1:]), pad_grad)))
        next_grad = F.deconv2d(pad_grad, self.input_shape, self.weight,
                               self.kernel_size, self.stride)
        return next_grad

    def gradient(self, learn_rate=1e-4):
        self.weight -= learn_rate * self.weight_grad
        self.bias -= learn_rate * self.bias_grad
        self.weight_grad = np.zeros(self.weight.shape)
        self.bias_grad = np.zeros(self.bias.shape)


class MaxPool2d(Layer):

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # python 实现，效率过低
        # N, C, H, W = x.shape
        # out = np.zeros([N, C, H // self.stride, W // self.stride])
        # self.mask = np.zeros(x.shape)
        # for batch in range(N):
        #     for channel in range(C):
        #         for i in range(0, H, self.stride):
        #             for j in range(0, W, self.stride):
        #                 out[batch, channel, i // self.stride,
        #                     j // self.stride] = np.max(
        #                         x[batch, channel, i:i + self.kernel_size,
        #                           j:j + self.kernel_size])
        #                 mask = np.argmax(x[batch, channel,
        #                                    i:i + self.kernel_size,
        #                                    j:j + self.kernel_size])
        #                 self.mask[batch, channel, i + mask // self.stride,
        #                           j + mask % self.stride] = 1
        out, self.mask = F.maxpool2d(x, self.kernel_size, self.stride)
        return out

    def backward(self, grad):
        return np.repeat(np.repeat(grad, self.stride, axis=2),
                         self.stride,
                         axis=3) * self.mask


class Linear(Layer):

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(in_features, out_features) / np.sqrt(
            in_features / 2)
        self.bias = np.random.randn(out_features) / np.sqrt(in_features / 2)

        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, x):  # x @ W.T + b.T
        self.x = x
        out = x @ self.weight + self.bias
        return out

    def backward(self, grad):
        for i in range(grad.shape[0]):
            # x[i][:, np.newaxis] 将一维向量转置 数学上等同于x.T 形式上等同于 x.reshape((-1, 1))
            self.weight_grad += self.x[i][:,
                                          np.newaxis] @ grad[i][np.newaxis, :]
            # x.T @ grad
            self.bias_grad += grad[i]
        new_grad = grad @ self.weight.T
        return new_grad

    def gradient(self, learn_rate=1e-4):
        self.weight -= learn_rate * self.weight_grad
        self.bias -= learn_rate * self.bias_grad

        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)


class Flatten(Layer):

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape((x.shape[0], -1))

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class ReLU(Layer):

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad):
        new_grad = grad
        new_grad[self.x < 0] = 0
        return new_grad


class Softmax(Layer):

    def forward(self, x):
        self.x = x
        return np.array(
            list(map(lambda vec: np.exp(vec) / np.sum(np.exp(vec), axis=1),
                     x)))

    def backward(self, grad):
        pass


class CrossEntropyLoss(Layer):

    def forward(self, x, label):
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
        self.x = x
        self.label = label
        for i in range(self.x.shape[0]):
            loss += -x[i, label[i]] + np.log(np.sum(np.exp(x[i])))
        return loss

    def backward(self):
        grad = np.array(
            list(map(lambda vec: np.exp(vec) / np.sum(np.exp(vec)), self.x)))
        for i in range(grad.shape[0]):
            grad[i, self.label[i]] -= 1
        return grad


def learning_rate_exponential_decay(learning_rate,
                                    global_step,
                                    decay_rate=0.1,
                                    decay_steps=3000):
    '''
    学习率指数衰减\n
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step/decay_steps)\n
    :return: learning rate decayed by step
    '''

    decayed_learning_rate = learning_rate * pow(
        decay_rate, float(global_step / decay_steps))
    return decayed_learning_rate


def train(model: Layer, criterion=CrossEntropyLoss(), epoachs=1):
    batch_size = 64
    loss = []
    for epoch in range(epoachs):
        for i in tqdm(range(train_img.shape[0] // batch_size)):
            lr = learning_rate_exponential_decay(0.1, epoch, 0.1, 10)
            # lr = 1e-4
            img = train_img[i * batch_size:(i + 1) * batch_size]
            label = train_lab[i * batch_size:(i + 1) * batch_size]
            output = model(img)
            loss.append(criterion(output, label))
            #print(loss)
            model.backward(criterion.backward())
            model.gradient(learn_rate=lr)
    return np.array(loss)


def valid(model):
    accu = 0
    batch_size = 64
    for i in tqdm(range(test_img.shape[0] // batch_size)):
        img = test_img[i * batch_size:(i + 1) * batch_size]
        label = test_lab[i * batch_size:(i + 1) * batch_size]
        output = model(img)
        accu += np.sum(np.argmax(output, axis=1) == label)
    return accu / test_img.shape[0]


if __name__ == '__main__':
    leNet = Sequence([
        Conv2d(1, 16, 3, padding=1),
        ReLU(),
        MaxPool2d(2, 2),
        Conv2d(16, 32, 5),
        ReLU(),
        MaxPool2d(2, 2),
        Flatten(),
        Linear(32*5*5, 120),
        ReLU(),
        Linear(120, 84),
        ReLU(),
        Linear(84, 10),
    ])
    # loss = train(leNet, epoachs=1)
    model = Sequence([
        Conv2d(1, 16, 5),
        MaxPool2d(2, 2),
        Conv2d(16, 16, 3, padding=1),
        ReLU(),
        Conv2d(16, 16, 3, padding=1),
        ReLU(),
        MaxPool2d(2, 2),
        Flatten(),
        Linear(16 * 6 * 6, 128),
        ReLU(),
        Linear(128, 10),
        # ReLU(),
        # Linear(84, 10),
    ])
    loss = train(model, epoachs=1)
    import code

    code.interact(local=locals())
