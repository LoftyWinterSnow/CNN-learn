from tqdm import tqdm
from base.Layer import *
from base.Variable import *
from base.Optimizer import *
import numpy as np
import matplotlib.pyplot as plt
import struct

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


def learning_rate_exponential_decay(learning_rate,
                                    global_step,
                                    decay_rate=0.1,
                                    decay_steps=5000):
    '''
    学习率指数衰减\n
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step/decay_steps)\n
    :return: learning rate decayed by step
    '''

    decayed_learning_rate = learning_rate * pow(
        decay_rate, float(global_step / decay_steps))
    return decayed_learning_rate


class ResNet(Layer):

    def __init__(self, input_var: Optional[Variable] = None):
        self.wait_backward = False
        self.initialized = False
        if not isinstance(input_var, type(None)):
            self.init_var(input_var.graph, input_var, Variable())

    def init_var(self, graph, input_var, output_var):
        if not self.initialized:
            self.input_var = input_var
            self.graph = graph
            out = Conv2d(1, 16, 5, input_var=input_var).output_var
            x1 = MaxPool2d(input_var=out).output_var
            out = Conv2d(16, 16, 3, input_var=x1, padding=1).output_var
            out = ReLU(input_var=out).output_var
            out = Conv2d(16, 16, 3, input_var=out, padding=1).output_var
            out = ReLU(input_var=out).output_var
            out = Add(x1, out).output_var
            out = MaxPool2d(input_var=out).output_var
            out = Flatten(input_var=out).output_var
            out = Linear(16 * 6 * 6, 128, input_var=out).output_var
            out = ReLU(input_var=out).output_var
            out = Linear(128, 10, input_var=out).output_var
            self.output_var = out
            self.initialized = True

    def _forward(self, x: np.ndarray):
        self.input_var.set_data(x)
        return self.output_var.eval(no_grad=True).data


def train(model: Layer, criterion=CrossEntropyLoss(), epoachs=1):
    batch_size = 64
    loss = []
    accu = []
    if model.initialized == False:
        img = Variable(name='img')
        label = Variable(graph=img.graph,
                         name='label',
                         dtype=np.uint8,
                         require_grad=False)
        model.init_var(img.graph, img,
                       Variable(graph=img.graph, name='prediction'))
        criterion.init_var(img.graph, [model.output_var, label],
                           Variable(graph=img.graph, name='loss'))
        optimizer = Adam(img.graph)
    else:
        optimizer = Adam(model.graph)
    # print(len(optimizer.params))
    for epoch in range(epoachs):
        for i in tqdm(range(train_img.shape[0] // batch_size)):
            lr = learning_rate_exponential_decay(1e-3, epoch)
            model.input_var.set_data(train_img[i * batch_size:(i + 1) *
                                               batch_size])
            criterion.input_var[1].set_data(train_lab[i * batch_size:(i + 1) *
                                                      batch_size])
            loss.append(criterion.output_var.eval().data)
            model.input_var.grad_eval()

            optimizer.gradient_descent(model.input_var.shape[0], lr=lr)
        accu.append(valid(model))

    return np.array(loss), np.array(accu)


def valid(model: Layer):
    batch_size = 64
    accu = 0
    for i in tqdm(range(test_img.shape[0] // batch_size)):
        img = test_img[i * batch_size:(i + 1) * batch_size]
        label = test_lab[i * batch_size:(i + 1) * batch_size]
        accu += np.sum(np.argmax(model(img), axis=1) == label)
    return accu / test_img.shape[0]


def show_loss_fig(loss, accu):

    fig = plt.figure(figsize=(13, 7), dpi=80)
    accu_new = [0] + list(accu)
    axis_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis_2 = axis_1.twinx()
    axis_1.plot(loss, color="#6AA84F")
    axis_2.plot(np.linspace(0, len(loss), len(accu_new)),
                accu_new,
                color="#5470C6")
    axis_1.set_xlabel('iteration')
    axis_1.set_ylabel('loss', color="#6AA84F")
    axis_2.set_ylabel('accuracy', color="#5470C6")
    plt.gca().spines["left"].set_color("#6AA84F")
    plt.gca().spines["right"].set_color("#5470C6")


def check_grad(graph: Graph):
    for i in graph.scope.keys():
        if 'grad' in i:
            print(i, np.sum(graph.scope[i].data))


if __name__ == '__main__':
    # model = Sequence([
    #     Flatten(),
    #     Linear(28 * 28, 128),
    #     ReLU(),
    #     Linear(128, 84),
    #     ReLU(),
    #     Linear(84, 10),
    # ])
    # loss,accu = train(model, epoachs=1)
    # leNet = Sequence([
    #     Conv2d(1, 6, 3, padding=1),
    #     ReLU(),
    #     MaxPool2d(2, 2),
    #     Conv2d(6, 16, 5),
    #     ReLU(),
    #     MaxPool2d(2, 2),
    #     Flatten(),
    #     Linear(16 * 5 * 5, 120),
    #     ReLU(),
    #     Linear(120, 84),
    #     ReLU(),
    #     Linear(84, 10),
    # ])
    # loss, accu = train(leNet, epoachs=5)
    loss, accu = train(ResNet(), epoachs=2)
    import code
    code.interact(local=locals())
