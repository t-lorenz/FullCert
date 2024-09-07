from typing import Tuple, List, Any

import torch
from torch import Tensor
from typeguard import typechecked

from boundflow.interval import Interval
from util import require


@typechecked
def correct_classifications(predictions: Interval, labels: Tensor) -> int:
    require(predictions.dim() == 2, f"Expecting predictions of shape [batch x logits] but got {predictions.size()}")
    require(labels.dim() == 1, f"Expecting labels of shape [batch] but got {labels.size()}")
    require(predictions.size(0) == labels.size(0), "predictions and labels must have the same batch size")
    if predictions.size(1) == 1:
        return __correct_binary_classifications(predictions, labels)
    else:
        return __correct_multi_classifications(predictions, labels)


@typechecked
def __correct_binary_classifications(predictions: Interval, labels: Tensor) -> int:
    mean_predictions = predictions.center().squeeze()
    one_hot_predictions = torch.where(mean_predictions >= 0,
                                      torch.ones_like(labels),
                                      torch.zeros_like(labels))
    return (one_hot_predictions == labels).int().sum().item()


@typechecked
def __correct_multi_classifications(predictions: Interval, labels: Tensor) -> int:
    mean_predictions = predictions.center()
    one_hot_predictions = torch.argmax(mean_predictions, dim=1)
    return (one_hot_predictions == labels).int().sum().item()


@typechecked
def certified_classifications(predictions: Interval, labels: Tensor) -> Tuple[int, int]:
    require(predictions.dim() == 2, f"Expecting predictions of shape [batch x logits] but got {predictions.size()}")
    require(labels.dim() == 1, f"Expecting labels of shape [batch] but got {labels.size()}")
    require(predictions.size(0) == labels.size(0), "predictions and labels must have the same batch size")
    if predictions.size(1) == 1:
        return __certified_binary_classifications(predictions, labels)
    else:
        return __certified_multi_classifications(predictions, labels)


@typechecked
def certified_classifications_ensemble(predictions: List[Interval], labels: Tensor) -> tuple[Any, Any]:
    num_classes = predictions[0].size(1)
    certified_classes = torch.full_like(labels, fill_value=-1)

    for cls in range(num_classes):
        violations_found = torch.full_like(certified_classes, dtype=torch.bool, fill_value=False)
        for logit in range(num_classes):
            for pred in range(len(predictions)):
                if cls != logit:
                    # noinspection PyTypeChecker
                    bounds_violated: Tensor = predictions[pred].upper[:, logit] >= predictions[pred].lower[:, cls]
                    violations_found = torch.logical_or(violations_found, bounds_violated)
        certified_class_i = torch.logical_not(violations_found)
        certified_classes[certified_class_i] = cls

    certified_robustness = (certified_classes >= 0).int().sum().item()
    certified_accuracy = (certified_classes == labels).int().sum().item()
    return certified_robustness, certified_accuracy


@typechecked
def __certified_multi_classifications(predictions: Interval, labels: Tensor) -> Tuple[int, int]:
    num_classes = predictions.size(1)
    certified_classes = torch.full_like(labels, fill_value=-1)

    for i in range(num_classes):
        violations_found = torch.full_like(certified_classes, dtype=torch.bool, fill_value=False)
        for j in range(num_classes):
            if i != j:
                # noinspection PyTypeChecker
                bounds_violated: Tensor = predictions.upper[:, j] >= predictions.lower[:, i]
                violations_found = torch.logical_or(violations_found, bounds_violated)
        certified_class_i = torch.logical_not(violations_found)
        certified_classes[certified_class_i] = i

    certified_robustness = (certified_classes >= 0).int().sum().item()
    certified_accuracy = (certified_classes == labels).int().sum().item()
    return certified_robustness, certified_accuracy


@typechecked
def __certified_binary_classifications(predictions: Interval, labels: Tensor) -> Tuple[int, int]:
    certified_classes = torch.full_like(labels, fill_value=-1)

    predictions = predictions.squeeze()

    certified_0 = predictions.upper < 0
    certified_1 = predictions.lower > 0

    certified_classes[certified_0] = 0
    certified_classes[certified_1] = 1

    certified_robustness = (certified_classes >= 0).int().sum().item()
    certified_accuracy = (certified_classes == labels).int().sum().item()
    return certified_robustness, certified_accuracy


@typechecked
def certified_binary_multi_model(predictions: List[Interval], labels: Tensor) -> Tensor:
    certified_aggregation = []
    for i in range(len(predictions)):

        prediction = predictions[i]

        certified_classes = torch.full_like(labels, fill_value=-1)

        prediction = prediction.squeeze()

        certified_0 = prediction.upper < 0
        certified_1 = prediction.lower > 0

        certified_classes[certified_0] = 0
        certified_classes[certified_1] = 1

        certified = (certified_classes == labels).int()
        certified_aggregation.append(certified)
    return torch.stack(certified_aggregation, dim=0).sum(dim=0)
