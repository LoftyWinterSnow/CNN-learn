import numpy as np
from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .Layer import Layer
    from .Variable import Variable


class Graph:

    def __init__(self):
        self.scope: Dict[str, Union[Variable, Layer]] = {}
        self.varNum = 0
        self.layerNum = 0

    def register(self, input_var: Union[List['Variable'], 'Variable'],
                 output_var: Union[List['Variable'],
                                   'Variable'], layer: 'Layer'):
        from .Variable import Variable
        self.scope[layer.name] = layer
        self.layerNum += 1
        if isinstance(input_var, Variable) and isinstance(
                output_var, Variable):
            input_var.child.append(layer.name)
            output_var.parent.append(layer.name)
            layer.parent.append(input_var.name)
            layer.child.append(output_var.name)

        elif isinstance(input_var, Variable) and len(output_var) > 1:
            for _output in output_var:
                _output.parent.append(layer.name)
                layer.child.append(_output.name)
            layer.parent.append(input_var.name)
            input_var.child.append(layer.name)

        elif isinstance(output_var, Variable) and len(input_var) > 1:
            for _input in input_var:
                _input.child.append(layer.name)
                layer.parent.append(_input.name)
            layer.child.append(output_var.name)
            output_var.parent.append(layer.name)

        elif len(output_var) > 1 and len(input_var) > 1:
            for _input in input_var:
                _input.child.append(layer.name)
                layer.parent.append(_input.name)
            for _output in output_var:
                _output.parent.append(layer.name)
                layer.child.append(_output.name)
