from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset
from typeguard import typechecked


@typechecked
def perturb_entire_dataset(dataset: Dataset, epsilon: float, data_range: Tuple[float, float]) -> Dataset:
    inputs = torch.stack([data[0] for data in dataset])
    labels = torch.stack([torch.tensor(data[1]) for data in dataset])
    lower_bounds = inputs - epsilon
    torch.clamp_min_(lower_bounds, min=data_range[0])
    upper_bounds = inputs + epsilon
    torch.clamp_max_(upper_bounds, max=data_range[1])
    return TensorDataset(lower_bounds, upper_bounds, labels)


@typechecked
def perturb_first_datapoint(dataset: Dataset, epsilon: float, data_range: Tuple[float, float]) -> Dataset:
    return perturb_one_datapoint(dataset=dataset, index_perturbed_point=0, epsilon=epsilon, data_range=data_range)


@typechecked
def perturb_single_pixels_per_dataset(dataset: Dataset, perturbed_pixels: int,
                                      data_range: Tuple[float, float]) -> Dataset:
    inputs = torch.stack([data[0] for data in dataset])
    labels = torch.stack([torch.tensor(data[1]) for data in dataset])
    flattened_inputs = inputs.view(-1)
    random_indices = torch.randint(low=0, high=len(flattened_inputs), size=(perturbed_pixels,))
    lower_bounds = flattened_inputs.clone()
    upper_bounds = flattened_inputs.clone()
    lower_bounds[random_indices] = data_range[0]
    upper_bounds[random_indices] = data_range[1]
    lower_bounds = lower_bounds.view(inputs.shape)
    upper_bounds = upper_bounds.view(inputs.shape)
    return TensorDataset(lower_bounds, upper_bounds, labels)


@typechecked
def perturb_single_pixels_per_image(dataset: Dataset, perturbed_pixels: int,
                                    data_range: Tuple[float, float]) -> Dataset:
    inputs = torch.stack([data[0] for data in dataset])
    labels = torch.stack([torch.tensor(data[1]) for data in dataset])
    flattened_inputs = inputs.view(inputs.shape[0], -1)
    random_indices = torch.randint(low=0, high=len(flattened_inputs[1]), size=(flattened_inputs[0], perturbed_pixels))
    lower_bounds = flattened_inputs.clone()
    upper_bounds = flattened_inputs.clone()
    lower_bounds[random_indices] = data_range[0]
    upper_bounds[random_indices] = data_range[1]
    lower_bounds = lower_bounds.view(inputs.shape)
    upper_bounds = upper_bounds.view(inputs.shape)
    return TensorDataset(lower_bounds, upper_bounds, labels)


@typechecked
def perturb_partition(inputs: torch.Tensor, labels: torch.Tensor, epsilon: float, data_range: Tuple[float, float],
                      partitions: int = 10, perturbed_partition: int = 0) -> Dataset:
    partition_size = inputs.shape[0] // partitions
    lower_bounds = inputs.clone()
    lower_bounds[perturbed_partition * partition_size:(perturbed_partition + 1) * partition_size] -= epsilon
    torch.clamp_min_(lower_bounds, min=data_range[0])
    upper_bounds = inputs.clone()
    upper_bounds[perturbed_partition * partition_size:(perturbed_partition + 1) * partition_size] += epsilon
    torch.clamp_max_(upper_bounds, max=data_range[1])
    return TensorDataset(lower_bounds, upper_bounds, labels)


@typechecked
def shuffle_data(data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(data.shape[0])
    return data[indices], labels[indices]


@typechecked
def perturb_one_datapoint(
        dataset: Dataset,
        index_perturbed_point: int,
        epsilon: float,
        data_range: Tuple[float, float]
) -> Dataset:
    inputs = torch.stack([data[0] for data in dataset])
    labels = torch.stack([torch.tensor(data[1]) for data in dataset])
    lower_bounds = torch.clone(inputs)
    upper_bounds = torch.clone(inputs)
    lower_bounds[index_perturbed_point] -= epsilon
    torch.clamp_min_(lower_bounds, min=data_range[0])
    upper_bounds[index_perturbed_point] += epsilon
    torch.clamp_max_(upper_bounds, max=data_range[1])
    return TensorDataset(lower_bounds, upper_bounds, labels)
