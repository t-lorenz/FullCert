from functools import partial
from typing import Union

import torch
from torch import Tensor
from typeguard import typechecked

from boundflow import interval
from boundflow.context import Context
from boundflow.interval import Interval, monotonic_bounds
from util import require


@typechecked
class Objective:

    def forward(
            self,
            logits:
            Union[Tensor, Interval],
            targets: Union[Tensor, Interval],
            context: Context
    ) -> Union[Tensor, Interval]:
        raise NotImplementedError()

    def backwards(self, context: Context) -> Union[Tensor, Interval]:
        raise NotImplementedError()


@typechecked
class CrossEntropyLoss(Objective):

    def forward(
            self,
            logits: Union[Tensor, Interval],
            targets: Union[Tensor, Interval],
            context: Context
    ) -> Union[Tensor, Interval]:
        require(len(targets.size()) == 1, "targets must be a 1D tensor")
        require(logits.size(0) == targets.size(0), "logits and targets must have the same size")
        context["logits"] = logits
        context["targets"] = targets
        batch_size = targets.size(0)
        # replaced with log softmax for better numerical stability
        # p = torch.softmax(logits, dim=1)
        # log_likelihood = -torch.log(p[range(batch_size), targets])
        log_softmax = interval.log_softmax(logits)
        log_likelihood = -log_softmax[range(batch_size), targets]
        loss = log_likelihood.mean()
        return loss

    def backwards(self, context: Context) -> Union[Tensor, Interval]:
        logits = context["logits"]
        targets = context["targets"]
        batch_size = targets.size(0)
        gradients = interval.softmax(logits)
        gradients[range(batch_size), targets] -= 1
        gradients = (1.0 / batch_size) * gradients
        return gradients


@typechecked
class BinaryCrossEntropyLoss(Objective):

    def forward(
            self,
            logits: Union[Tensor, Interval],
            targets: Union[Tensor, Interval],
            context: Context
    ) -> Union[Tensor, Interval]:
        require(len(targets.size()) == 1, "targets must be a 1D tensor")
        require(logits.size(0) == targets.size(0), "logits and targets must have the same size")
        targets = targets.float()
        logits = logits.squeeze()
        context["logits"] = logits
        context["targets"] = targets
        loss = interval.monotonic_bounds(logits, lambda x: torch.binary_cross_entropy_with_logits(x, targets))
        return loss.mean()

    def backwards(self, context: Context) -> Union[Tensor, Interval]:
        logits = context["logits"]
        targets = context["targets"]
        gradients = interval.monotonic_bounds(logits, lambda x: torch.sigmoid(x) - targets)
        return ((1.0 / targets.size(0)) * gradients).unsqueeze(1)


@typechecked
class HingeLoss(Objective):

    def forward(
            self,
            logits: Union[Tensor, Interval],
            targets: Tensor,
            context: Context
    ) -> Union[Tensor, Interval]:
        require(len(targets.size()) == 1, "targets must be a 1D tensor")
        require(torch.all(torch.logical_or(targets == 0, targets == 1)), "targets must be 0 or 1")
        require(logits.size(0) == targets.size(0), "logits and targets must have the same size")
        targets = targets.unsqueeze(1)
        context["logits"] = logits
        context["targets"] = targets
        # change targets to -1 and 1
        targets = 2.0 * targets - 1.0
        individual_losses = monotonic_bounds(logits, partial(HingeLoss.__hinge, targets=targets))
        loss = individual_losses.sum(dim=0)
        return loss

    @staticmethod
    def __hinge(logits: Tensor, targets: Tensor) -> Tensor:
        return torch.max(torch.zeros_like(logits), logits * targets)

    def backwards(self, context: Context) -> Union[Tensor, Interval]:
        logits: Union[torch.Tensor, Interval] = context["logits"]
        targets: torch.Tensor = context["targets"]
        require(logits.size() == targets.size(),
                f"logits and targets must have the same size! {logits.size()} != {targets.size()}")
        require(torch.all(torch.logical_or(targets == 0, targets == 1)),
                "targets must be 0 or 1")
        # change targets to -1 and 1
        targets = 2.0 * targets - 1.0
        batch_size = targets.size(0)
        gradients = monotonic_bounds(logits, partial(HingeLoss.__gradients, targets=targets))
        gradients = (1 / batch_size) * gradients
        return gradients

    @staticmethod
    def __gradients(logits: Tensor, targets: Tensor) -> Tensor:
        return torch.where(targets * logits < 1.0, -targets, torch.zeros_like(logits))
