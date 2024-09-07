from typing import Union

import torch
from typeguard import typechecked


@typechecked
def require(condition: Union[bool, torch.Tensor], message: str = "") -> None:
    condition = condition if isinstance(condition, bool) else condition.item()
    if not condition:
        raise ValueError(message)
