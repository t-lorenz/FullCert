import math
from typing import Union, List, Tuple

import torch
from torch import Tensor
from torch.nn import init
from typeguard import typechecked

from boundflow.context import Context
from boundflow.interval import Interval
from util import require


@typechecked
class Layer:

    def forward(self, x: Union[Interval, Tensor], context: Context) -> Union[Tensor, Interval]:
        raise NotImplementedError()

    def backwards(self, outer_grad: Union[Interval, Tensor], context: Context) \
            -> Union[Tuple[Tensor, List[Tensor]], Tuple[Interval, List[Interval]]]:
        raise NotImplementedError()

    def params(self) -> Union[List[Tensor], List[Interval]]:
        raise NotImplementedError()

    def update_weights(self, gradients: Union[List[Interval], List[Tensor]], learning_rate: float) -> None:
        raise NotImplementedError()

    def state_dict(self) -> dict:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError()

    def clone(self) -> "Layer":
        raise NotImplementedError()

    def freeze(self, value: bool) -> None:
        raise NotImplementedError()


@typechecked
class LinearLayer(Layer):

    def __init__(self, in_features: int, out_features: int):
        super(LinearLayer, self).__init__()
        self.weights = torch.empty(in_features, out_features)
        self.biases = torch.empty(1, out_features)
        self.reset_parameters()
        self.in_features = in_features
        self.out_features = out_features
        self.frozen = False

    # Pytorch default initialization of linear layers
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L92
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.biases is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.biases, -bound, bound)

    def params(self) -> Union[List[Tensor], List[Interval]]:
        return [self.weights, self.biases]

    def forward(self, x: Union[Tensor, Interval], context: Context) -> Union[Tensor, Interval]:
        require(len(x.size()) == 2, "Expecting 2 dimensional tensor [batch x features]")
        require(x.size(1) == self.in_features, f"Expected feature dimension is {self.in_features}, but got {x.size(1)}")
        context["weights"] = self.weights
        context["biases"] = self.biases
        context["x"] = x
        return x @ self.weights + self.biases

    def backwards(self, grad_output: Union[Tensor, Interval], context: Context) \
            -> Union[Tuple[Tensor, List[Tensor]], Tuple[Interval, List[Interval]]]:
        x, weights = context["x"], context["weights"]
        m = x.size(0)
        d_x = grad_output @ weights.t()
        d_weight = (1 / m) * (x.t() @ grad_output)
        d_bias = (1 / m) * grad_output.sum(dim=0, keepdim=True)
        return d_x, [d_weight, d_bias]

    def update_weights(self, gradients: Union[List[Tensor], List[Interval]], learning_rate: float) -> None:
        require(len(gradients) == 2, "Expecting 2 gradients for weights and biases")
        if not self.frozen:
            self.weights = self.weights - gradients[0] * learning_rate
            self.biases = self.biases - gradients[1] * learning_rate

    def freeze(self, value: bool) -> None:
        self.frozen = value

    def clone(self) -> "LinearLayer":
        layer = LinearLayer(self.in_features, self.out_features)
        layer.weights = self.weights.clone()
        layer.biases = self.biases.clone()
        return layer

    def state_dict(self) -> dict:
        return {"weights": self.weights, "biases": self.biases}

    def load_state_dict(self, state_dict: dict) -> None:
        self.weights = state_dict["weights"]
        self.biases = state_dict["biases"]

    def __repr__(self) -> str:
        # return f"LinearLayer(in_features={self.in_features}, out_features={self.out_features}, " \
        #        f"weights=Tensor({self.weights}), biases=Tensor({self.biases}))"
        return f"LinearLayer(in_features={self.in_features}, out_features={self.out_features}, " \
               f"weights=Tensor({self.weights.size()}), biases=Tensor({self.biases.size()}))"


@typechecked
class ReLU(Layer):

    def forward(self, x: Union[Tensor, Interval], context: Context) -> Union[Tensor, Interval]:
        context["x"] = x
        if isinstance(x, Interval):
            return Interval(lower=torch.relu(x.lower), upper=torch.relu(x.upper))
        else:
            return torch.relu(x)

    def backwards(self, grad_output: Union[Tensor, Interval], context: Context) \
            -> Union[Tuple[Tensor, List[Tensor]], Tuple[Interval, List[Interval]]]:
        x = context["x"]
        if isinstance(x, Interval):
            gradient_upper = torch.where(x.upper > 0.0, 1.0, 0.0)
            gradient_lower = torch.where(x.lower > 0.0, 1.0, 0.0)
            gradient = Interval(lower=gradient_lower, upper=gradient_upper)
        else:
            gradient = torch.where(x > 0.0, 1.0, 0.0)
        return gradient * grad_output, []

    def params(self) -> List[Union[Tensor, Interval]]:
        return []

    def update_weights(self, gradients: Union[List[Tensor], List[Interval]], learning_rate: float) -> None:
        require(len(gradients) == 0, "Expecting no gradients for ReLU")

    def freeze(self, value: bool) -> None:
        pass  # No parameters to freeze

    def clone(self) -> "ReLU":
        return ReLU()

    def state_dict(self) -> dict:
        return {}  # no state to store

    def load_state_dict(self, state_dict: dict) -> None:
        pass  # no state to load

    def __repr__(self):
        return "ReLU"
