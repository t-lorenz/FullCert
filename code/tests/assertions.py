from typing import Tuple

import torch
from torch import Tensor
from typeguard import typechecked

from boundflow.interval import Interval


@typechecked
def almost_leq(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool(torch.all(torch.logical_or(torch.isclose(x, y), x <= y)))


@typechecked
def almost_geq(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool(torch.all(torch.logical_or(torch.isclose(x, y), x >= y)))


@typechecked
def almost_contains(x: Interval, y: torch.Tensor) -> bool:
    return almost_leq(x.lower, y) and almost_geq(x.upper, y)


@typechecked
def uniform(size: Tuple[int, ...], lower: float = 0.0, upper: float = 1.0) -> Tensor:
    return torch.rand(size=size) * (upper - lower) + lower


@typechecked
def uniform_target(size: int, num_classes: int) -> Tensor:
    return torch.randint(low=0, high=num_classes, size=(size,))


@typechecked
def uniform_interval(size: Tuple[int, ...], lower: float = 0.0, upper: float = 1.0) -> Interval:
    center = uniform(size=size, lower=lower, upper=upper)
    radius = uniform(size=size, lower=0, upper=(upper - lower) / 2.0)
    return Interval.from_center_radius(center, radius)


@typechecked
def uniform_from(interval: Interval) -> torch.Tensor:
    return torch.rand_like(interval.lower) * interval.radius() * 2.0 + interval.lower
