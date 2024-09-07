from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import torch.utils.data
import torchvision
from sklearn import datasets
from torch.utils.data import Dataset, TensorDataset, Subset
from torch.utils.data import random_split
from typeguard import typechecked

from util import require

MNIST_DIR = "../data/mnist"
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


@typechecked
class DataContainer:
    def __init__(
            self,
            train_set: Dataset,
            val_set: Dataset,
            test_set: Dataset,
            pretrain_set: Dataset,
            feature_dim: List[int],
            label_dim: int,
            data_range: Tuple[float, float]
    ):
        self.train_set: Dataset = train_set
        self.val_set: Dataset = val_set
        self.test_set: Dataset = test_set
        self.pretrain_set: Dataset = pretrain_set
        self.feature_dim: List[int] = feature_dim
        self.label_dim: int = label_dim
        self.data_range: Tuple[float, float] = data_range


@typechecked
class DatasetType(Enum):
    MNIST = "mnist"
    MOONS = "moons"
    MNIST_1_7 = "mnist17"

    def __str__(self) -> str:
        return self.value


@typechecked
def load_dataset(dataset: DatasetType, pretrain_size: int, train_size: Union[int, None], test_size: Union[int, None],
                 **kwargs) -> DataContainer:
    if dataset == DatasetType.MNIST:
        return load_mnist(pretrain_size=pretrain_size, train_size=train_size, standardize=False, **kwargs)
    elif dataset == DatasetType.MNIST_1_7:
        return load_mnist_1_7(pretrain_size=pretrain_size, train_size=train_size, test_size=test_size,
                              standardize=False, **kwargs)
    elif dataset == DatasetType.MOONS:
        return load_moons(pretrain_size=pretrain_size, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")


@typechecked
def load_mnist(val_size=5_000, pretrain_size=40, train_size: Union[int, None] = None, standardize=True,
               resize=28) -> DataContainer:
    if standardize:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                MNIST_MEAN, MNIST_STD)
        ])
        data_range = ((0.0 - MNIST_MEAN[0]) / MNIST_STD[0], (1.0 - MNIST_MEAN[0]) / MNIST_STD[0])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.ToTensor()
        ])
        data_range = (0.0, 1.0)
    train_set = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True, transform=transform)
    dataset_size = len(train_set)
    max_train_size = dataset_size - val_size - pretrain_size
    require(train_size is None or max_train_size >= train_size, "train_size is too large")
    if train_size is None:
        train_size = max_train_size
    train_set, val_set, pretrain_set, _ = random_split(train_set, (
        train_size, val_size, pretrain_size, dataset_size - train_size - val_size - pretrain_size))
    test_set = torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True, transform=transform)
    return DataContainer(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        pretrain_set=pretrain_set,
        feature_dim=[1, resize, resize],
        label_dim=10,
        data_range=data_range
    )


@typechecked
def load_mnist_1_7(val_size=2_000, pretrain_size=10, train_size: Union[int, None] = None,
                   test_size: Union[int, None] = None, standardize=True,
                   resize=28) -> DataContainer:
    if standardize:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                MNIST_MEAN, MNIST_STD)
        ])
        data_range = ((0.0 - MNIST_MEAN[0]) / MNIST_STD[0], (1.0 - MNIST_MEAN[0]) / MNIST_STD[0])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.ToTensor()
        ])
        data_range = (0.0, 1.0)
    train_set = torchvision.datasets.MNIST(MNIST_DIR, train=True, download=True, transform=transform)
    train_set = __filter_dataset(train_set)
    # noinspection PyTypeChecker
    dataset_size = len(train_set)
    max_train_size = dataset_size - val_size - pretrain_size
    require(train_size is None or max_train_size >= train_size, "train_size is too large")
    if train_size is None:
        train_size = max_train_size
    train_set, val_set, pretrain_set, _ = random_split(train_set, (
        train_size, val_size, pretrain_size, dataset_size - train_size - val_size - pretrain_size))
    test_set = torchvision.datasets.MNIST(MNIST_DIR, train=False, download=True, transform=transform)
    test_set = __filter_dataset(test_set)
    if test_size is not None:
        test_set, _ = random_split(test_set, (test_size, len(test_set) - test_size))
    return DataContainer(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        pretrain_set=pretrain_set,
        feature_dim=[1, resize, resize],
        label_dim=2,
        data_range=data_range
    )


@typechecked
def __filter_dataset(dataset: Dataset) -> Dataset:
    inputs = torch.stack([data[0] for data in dataset])
    labels = torch.stack([torch.tensor(data[1]) for data in dataset])
    mask = torch.logical_or(labels == 6, labels == 0)
    labels[labels == 6] = 1
    return TensorDataset(inputs[mask], labels[mask])


@typechecked
def load_moons(train_size: int = 1_000, val_size: int = 200, pretrain_size: int = 8) -> DataContainer:
    return DataContainer(
        train_set=__make_moons(train_size),
        val_set=__make_moons(val_size),
        test_set=__make_moons(val_size),
        pretrain_set=__make_moons(pretrain_size),
        feature_dim=[2],
        label_dim=2,
        data_range=(float("-inf"), float("inf"))
    )


@typechecked
def __make_moons(samples: int) -> Dataset:
    features, labels = datasets.make_moons(n_samples=samples, shuffle=True, noise=0.1)
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)
    return torch.utils.data.TensorDataset(features, labels)


@typechecked
def as_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.stack([data[0] for data in dataset]), torch.stack([torch.tensor(data[1]) for data in dataset])


@typechecked
def create_random_subset(dataset: Dataset, subset_size: int) -> Dataset:
    # noinspection PyTypeChecker
    dataset_size = len(dataset)
    require(subset_size <= dataset_size, "The subset cannot be larger than the original dataset")
    subset_indices = np.random.choice(dataset_size, size=subset_size, replace=False)
    return Subset(dataset, subset_indices)
