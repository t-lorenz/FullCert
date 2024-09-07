from typing import Union

from torch import Tensor
from typeguard import typechecked

from boundflow.interval import Interval


@typechecked
class Context:

    def __init__(self):
        self.data = {}

    def __setitem__(self, key: str, value: Union[Tensor, Interval]) -> None:
        self.data[key] = value.clone()

    def __getitem__(self, key: str) -> Union[Tensor, Interval]:
        return self.data[key]
