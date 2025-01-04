import numpy as np
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from .Graph import Graph


class Variable:

    def __init__(self,
                 data: Union[list, np.ndarray, None] = None,
                 dtype=np.double,
                 require_grad: bool = True,
                 learnable: bool = False,
                 name: Optional[str] = None,
                 graph: Optional[Graph] = None
                 ):
        # 方便编译器识别变量
        if not isinstance(data, type(None)):
            self.data = np.asarray(data, dtype=dtype)
            self.hasValue = True
        else:
            self.hasValue = False
        self.graph = graph
        if self.graph is None:
            self.graph = Graph()
        self.dtype = dtype
        self.requires_grad = require_grad
        self.learnable = learnable
        self.parent = []
        self.child = []

        self.name = name
        if self.name is None:
            self.name = "Var" + str(self.graph.varNum)

        self.graph.varNum += 1
        if self.name in self.graph.scope:
            raise ValueError("Variable name already exists")
        self.graph.scope[self.name] = self

        if self.requires_grad:
            if self.hasValue:
                self.grad = Variable(np.zeros_like(self.data, dtype=np.double),
                                     require_grad=False,
                                     name=self.name + '_grad',
                                     graph=self.graph)
            else:
                self.grad = Variable(require_grad=False,
                                     name=self.name + '_grad',
                                     graph=self.graph)

        self.wait_backward = False

    @property
    def shape(self) -> Union[Tuple[int, ...], None]:
        if self.hasValue:
            return self.data.shape
        else:
            return None

    def __repr__(self):
        if self.hasValue:
            return repr(self.data).replace('array', 'Variable')
        else:
            return 'Variable(None)'

    def eval(self, no_grad = False):
        if self.wait_backward:
            pass
        else:
            for layer in self.parent:
                self.graph.scope[layer].forward(no_grad=no_grad)
            if not no_grad:
                self.wait_backward = True
        return self

    def grad_eval(self, no_grad = False):
        if self.requires_grad:
            if self.wait_backward:
                for layer in self.child:
                    self.graph.scope[layer].backward(no_grad=no_grad)
                if not no_grad:
                    self.wait_backward = False
            else:
                pass
            return self.grad
        else:
            return None

    def set_data(self, data):
        if self.hasValue:
            self.data = np.asarray(data, dtype=self.dtype)
        else:
            self.data = np.asarray(data, dtype=self.dtype)
            if self.requires_grad:
                self.grad.set_data(np.zeros_like(self.data, dtype=np.double))
            self.hasValue = True
    
    def add_data(self, data):
        if self.hasValue:
            self.data += np.asarray(data, dtype=self.dtype)
        else:
            self.set_data(data)


if __name__ == '__main__':
    a = Variable([[1, 2, 3]], name='x', learnable=True)

    import code
    code.interact(local=locals())
