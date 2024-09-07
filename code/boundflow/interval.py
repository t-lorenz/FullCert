from typing import Iterable, Union, Tuple, Callable

import torch
from torch import Tensor
from typeguard import typechecked

from util import require


@typechecked
class Interval:

    @staticmethod
    def empty_interval(center: Tensor) -> "Interval":
        return Interval(lower=center.clone(), upper=center)

    @staticmethod
    def from_center_radius(center: Tensor, radius: Tensor) -> "Interval":
        return Interval(lower=center - radius, upper=center + radius)

    def __init__(self, lower: Tensor, upper: Tensor):
        require(torch.all(torch.isfinite(lower)), "Lower bound contains non-finite numbers!")
        require(torch.all(torch.isfinite(upper)), "Upper bound contains non-finite numbers!")
        require(torch.all((upper - lower) > -1e8), f"lower bound has to be smaller or equal to upper bound. "
                                                   f"Violations: {torch.sum(lower > upper)}. "
                                                   f"Max difference: {torch.max(lower - upper)}")
        self.lower: Tensor = lower
        self.upper: Tensor = upper

    def center(self) -> Tensor:
        return (self.lower + self.upper) / 2.0

    def flatten(self) -> Tensor:
        return self.center().flatten()

    def squeeze(self) -> "Interval":
        return Interval(lower=self.lower.squeeze(), upper=self.upper.squeeze())

    def unsqueeze(self, dim: int) -> "Interval":
        return Interval(lower=self.lower.unsqueeze(dim), upper=self.upper.unsqueeze(dim))

    def radius(self) -> Tensor:
        return (self.upper - self.lower) / 2.0

    def __contains__(self, x: Tensor) -> bool:
        return bool(torch.all(self.lower <= x) and torch.all(x <= self.upper))

    def __repr__(self) -> str:
        return f"Interval(lower={self.lower}, upper={self.upper}"

    def __add__(self, other) -> "Interval":
        if isinstance(other, Interval):
            return Interval(
                lower=self.lower + other.lower,
                upper=self.upper + other.upper
            )
        else:
            return Interval(
                lower=self.lower + other,
                upper=self.upper + other
            )

    __radd__ = __add__

    def __sub__(self, other) -> "Interval":
        if isinstance(other, Interval):
            return Interval(
                lower=self.lower - other.upper,
                upper=self.upper - other.lower
            )
        else:
            return Interval(
                lower=self.lower - other,
                upper=self.upper - other
            )

    def __rsub__(self, other) -> "Interval":
        return Interval(
            lower=other - self.upper,
            upper=other - self.lower
        )

    def __neg__(self) -> "Interval":
        return Interval(lower=-self.upper, upper=-self.lower)

    def __mul__(self, other) -> "Interval":
        if isinstance(other, Interval):
            return Interval(
                lower=torch.min(
                    input=torch.stack([
                        self.lower * other.lower,
                        self.lower * other.upper,
                        self.upper * other.lower,
                        self.upper * other.upper
                    ]),
                    dim=0
                )[0],
                upper=torch.max(
                    input=torch.stack([
                        self.lower * other.lower,
                        self.lower * other.upper,
                        self.upper * other.lower,
                        self.upper * other.upper
                    ]),
                    dim=0
                )[0]
            )
        else:
            return Interval(
                lower=torch.min(self.lower * other, self.upper * other),
                upper=torch.max(self.lower * other, self.upper * other)
            )

    __rmul__ = __mul__

    def __matmul__(self, other) -> "Interval":
        # Implementation based on Rump's algorithm https://link.springer.com/article/10.1023/A:1022374804152
        # More precise, but also more expensive: https://hal.inria.fr/inria-00469472/document. Requires rounding
        # numerical errors up and down, which we currently don't support.
        require(len(other.size()) == len(self.size()) == 2,
                f"Expecting two-dimensional matrices, got {self.size()} and {other.size()}")
        require(self.size(1) == other.size(0),
                f"Dimension mismatch! Got {self.size()} and {other.size()}")
        center_a, radius_a = self.center(), self.radius()
        if isinstance(other, Interval):
            center_b, radius_b = other.center(), other.radius()
            return Interval.from_center_radius(
                center=center_a @ center_b,
                radius=(abs(center_a) + radius_a) @ radius_b + radius_a @ abs(center_b)
            )
        else:
            return Interval.from_center_radius(
                center=center_a @ other,
                radius=radius_a @ abs(other)
            )

    def __rmatmul__(self, other) -> "Interval":
        require(len(other.size()) == len(self.size()) == 2,
                f"Expecting two-dimensional matrices, got {self.size()} and {other.size()}")
        require(other.size(1) == self.size(0),
                f"Dimension mismatch! Got {self.size()} and {other.size()}")
        return Interval.from_center_radius(
            center=other @ self.center(),
            radius=abs(other) @ self.radius()
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, Interval):
            return self.lower == other.lower and self.upper == other.upper
        return False

    def __getitem__(self, item) -> "Interval":
        return Interval(self.lower.__getitem__(item), self.upper.__getitem__(item))

    def __setitem__(self, key, value):
        if isinstance(value, Interval):
            self.lower.__setitem__(key, value.lower)
            self.upper.__setitem__(key, value.upper)
        else:
            self.lower.__setitem__(key, value)
            self.upper.__setitem__(key, value)

    def __len__(self) -> int:
        return self.lower.__len__()

    # Make sure __radd__ etc work correctly with numpy ndarray
    __numpy_ufunc__ = None  # Numpy up to 13.0
    __array_ufunc__ = None  # Numpy 13.0 and above

    def t(self) -> "Interval":
        return Interval(lower=self.lower.t(), upper=self.upper.t())

    def clone(self) -> "Interval":
        return Interval(lower=self.lower.clone(), upper=self.upper.clone())

    def size(self, dim: int = None) -> Union[torch.Size, int]:
        if dim is None:
            return self.lower.size()
        else:
            return self.lower.size(dim)

    def dim(self) -> int:
        return self.lower.dim()

    def sum(self, dim=None, keepdim: bool = False) -> "Interval":
        if dim is None:
            return Interval(
                lower=self.lower.sum(),
                upper=self.upper.sum()
            )
        else:
            return Interval(
                lower=self.lower.sum(dim=dim, keepdim=keepdim),
                upper=self.upper.sum(dim=dim, keepdim=keepdim)
            )

    def mean(self, dim=None, keepdim: bool = False) -> "Interval":
        if dim is None:
            return Interval(
                lower=self.lower.mean(),
                upper=self.upper.mean()
            )
        else:
            return Interval(
                lower=self.lower.mean(dim=dim, keepdim=keepdim),
                upper=self.upper.mean(dim=dim, keepdim=keepdim)
            )

    def random_point(self) -> Tensor:
        return torch.rand_like(self.upper) * (self.upper - self.lower) + self.lower


@typechecked
def square(x: Union[Interval, float, Tensor]) -> Union[Interval, float, Tensor]:
    if isinstance(x, Interval):
        # Default case with positive and negative ranges
        lower = torch.zeros_like(x.lower)
        upper = torch.maximum(torch.square(x.lower), torch.square(x.upper))
        # Special case where all values are positive
        lower = torch.where(
            x.lower >= 0,
            torch.square(x.lower),
            lower
        )
        upper = torch.where(
            x.lower >= 0,
            torch.square(x.upper),
            upper
        )
        # Special case where all values are negative
        lower = torch.where(
            x.upper <= 0,
            torch.square(x.upper),
            lower
        )
        upper = torch.where(
            x.upper <= 0,
            torch.square(x.lower),
            upper
        )
        return Interval(lower, upper)
    else:
        return torch.square(x)


@typechecked
def softmax(x: Union[float, Interval, Tensor]) -> Union[Interval, Tensor]:
    require(x.dim() == 2, "expecting 2D tensor with dimensions [batch, logits]")
    if isinstance(x, Interval):
        lower_bound = __softmax_bound(x.lower, x.upper)
        upper_bound = __softmax_bound(x.upper, x.lower)
        return Interval(lower=lower_bound, upper=upper_bound)
    else:
        return torch.softmax(x, dim=1)


@typechecked
def __softmax_bound(current_bound: Tensor, opposite_bound: Tensor) -> Tensor:
    # numerically stable implementation of tight upper and lower bounds
    # computes one of the bounds. current_bound is upper if calculating upper bounds, lower if calculating the lower
    # bound. opposite_bound is lower if currently computing the upper bound, and vice versa.
    size = current_bound.size(1)

    # components contain the elements required for the summation in the denominator. That is the bound for position i
    # and opposite bound for all j != i
    components = opposite_bound.unsqueeze(1).repeat(1, size, 1)
    components[..., range(size), range(size)] = current_bound
    # offsets for computationally stable exp(). We use a different value for each logit dimension, since the sum entries
    # differ due to using different bounds.
    offsets = torch.max(components, dim=-1)[0]
    # numerator for each dimension
    numerator = torch.exp(current_bound - offsets)
    # denominator for each dimension. Note that in contrast to traditional implementation, the sums differ across
    # logit dimensions due to different upper / lower bounds (and also different offsets)
    denominator = torch.sum(torch.exp(components - offsets[..., None]), dim=-1)
    return numerator / denominator


@typechecked
def log_softmax(x: Union[Interval, Tensor]) -> Union[Interval, Tensor]:
    require(x.dim() == 2, "expecting 2D tensor with dimensions [batch, logits]")
    if isinstance(x, Interval):
        lower_bound = __log_softmax_bound(x.lower, x.upper)
        upper_bound = __log_softmax_bound(x.upper, x.lower)
        return Interval(lower=lower_bound, upper=upper_bound)
    else:
        return torch.log_softmax(x, dim=1)


@typechecked
def __log_softmax_bound(current_bound: Tensor, opposite_bound: Tensor) -> Tensor:
    # numerically stable implementation of tight upper and lower bounds
    # computes one of the bounds. current_bound is upper if calculating upper bounds, lower if calculating the lower
    # bound. opposite_bound is lower if currently computing the upper bound, and vice versa.
    size = current_bound.size(1)

    # components contain the elements required for the summation in the denominator. That is the bound for position i
    # and opposite bound for all j != i
    components = opposite_bound.unsqueeze(1).repeat(1, size, 1)
    components[..., range(size), range(size)] = current_bound
    # offsets for computationally stable exp(). We use a different value for each logit dimension, since the sum entries
    # differ due to using different bounds.
    offsets = torch.max(components, dim=-1)[0]
    # first component for each component
    first_component = current_bound - offsets
    # sum component for each dimension. Note that in contrast to traditional implementation, the sums differ across
    # logit dimensions due to different upper / lower bounds (and also different offsets)
    sum_component = torch.sum(torch.exp(components - offsets[..., None]), dim=-1)
    return first_component - torch.log(sum_component)


@typechecked
def log(x: Union[Interval, Tensor]) -> Union[Interval, Tensor]:
    if isinstance(x, Interval):
        return Interval(lower=torch.log(x.lower), upper=torch.log(x.upper))
    else:
        return torch.log(x)


@typechecked
def exp(x: Union[Interval, Tensor]) -> Union[Interval, Tensor]:
    if isinstance(x, Interval):
        return Interval(lower=torch.exp(x.lower), upper=torch.exp(x.upper))
    else:
        return torch.exp(x)


@typechecked()
def sigmoid(x: Union[Interval, Tensor]) -> Union[Interval, Tensor]:
    return monotonic_bounds(x, torch.sigmoid)


@typechecked
def stack(elements: Union[Iterable[Interval], Iterable[Tensor]], axis: int) -> Union[Interval, Tensor]:
    if all(isinstance(it, Interval) for it in elements):
        return Interval(
            lower=torch.stack([element.lower for element in elements], axis),
            upper=torch.stack([element.upper for element in elements], axis)
        )
    else:
        return torch.stack(elements, axis)


@typechecked
def uniform(size: Tuple[int, ...], lower: float = 0.0, upper: float = 1.0) -> Tensor:
    return torch.rand(size=size) * (upper - lower) + lower


@typechecked
def monotonic_bounds(x: Union[Interval, Tensor], function: Callable[[Tensor], Tensor]) -> Union[Interval, Tensor]:
    if isinstance(x, Interval):
        return Interval(
            lower=torch.min(function(x.lower), function(x.upper)),
            upper=torch.max(function(x.lower), function(x.upper))
        )
    else:
        return function(x)
