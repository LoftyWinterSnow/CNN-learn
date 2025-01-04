# 简介

本程序旨在不依赖现有的深度学习框架，如pytorch, tensorflow, keras等，实现一个具有卷积层，池化层，全连接层，损失函数，激活函数的神经网络框架。

# 环境

python版本选用3.9.19，包版本见requirements.txt，使用

`pip install -r requirements.txt`

即可安装。

feature_map.dll依赖

`libgcc_s_seh-1.dll`

`KERNEL32.dll `

`msvcrt.dll `

`USER32.dll `

`libgomp-1.dll `

`libstdc++-6.dll`

请确保这些库文件存在并能被正确识别

# 使用

使用模板见`cnn_new.py`，使用方式与pytorch接近。

使用`Sequence`类可以快速的创建网络，示例如下:

```
model = Sequence([
    Flatten(),
    Linear(28 * 28, 128),
    ReLU(),
    Linear(128, 84),
    ReLU(),
    Linear(84, 10),
])
```

或者通过继承`Layer`类来创建复杂网络

```python
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
```

在`init_var()`函数中可以定义网络结构，然后重载`__init__()`函数和`_forward()`函数即完成定义
