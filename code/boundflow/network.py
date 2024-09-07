from typing import Union, List, Tuple

from torch import Tensor
from typeguard import typechecked

from boundflow.context import Context
from boundflow.interval import Interval
from boundflow.layers import Layer
from util import require


@typechecked
class Model:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.contexts = []

    def forward(self, x: Union[Tensor, Interval]) -> Union[Tensor, Interval]:
        out = x
        for layer in self.layers:
            context = Context()
            self.contexts.append(context)
            out = layer.forward(out, context)
        return out

    def backwards(self, grad_output: Union[Tensor, Interval]) \
            -> Union[Tuple[Interval, List[List[Interval]]], Tuple[Interval, List[List[Interval]]]]:
        layer_gradients = []
        previous_grad = grad_output
        for (layer, context) in zip(reversed(self.layers), reversed(self.contexts)):
            grads = layer.backwards(previous_grad, context)
            previous_grad = grads[0]
            layer_gradients.append(grads[1])
        layer_gradients.reverse()
        return previous_grad, layer_gradients

    def update_weights(self, layer_gradients: Union[List[List[Tensor]], List[List[Interval]]], learning_rate: float):
        require(len(layer_gradients) == len(self.layers), "expected exactly one gradient list per layer")
        for gradients, layer in zip(layer_gradients, self.layers):
            layer.update_weights(gradients, learning_rate)

    def clone(self) -> "Model":
        return Model([layer.clone() for layer in self.layers])

    def state_dict(self) -> dict:
        return {
            "layers": [layer.state_dict() for layer in self.layers]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for layer, layer_state_dict in zip(self.layers, state_dict["layers"]):
            layer.load_state_dict(layer_state_dict)

    def __repr__(self) -> str:
        return f"Model(layers={self.layers})"
