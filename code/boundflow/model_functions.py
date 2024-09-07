import time
from typing import Tuple, List

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from typeguard import typechecked

from boundflow.context import Context
from boundflow.interval import Interval
from boundflow.metrics import correct_classifications, certified_classifications, certified_classifications_ensemble

from boundflow.network import Model
from boundflow.objective import Objective
from util import require

optimal_weights = torch.tensor([[-0.0913, 1.0354], [0.9888, -1.1611]])
optimal_bias = torch.tensor([[-0.3462, -0.3980]])


@typechecked
def train_full_batch(
        model: Model,
        epochs: int,
        train_set: Dataset,
        val_set: Dataset,
        learning_rate: float,
        loss_function: Objective,
) -> float:
    pbar = trange(epochs)

    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    max_certified_accuracy = 0.0

    for epoch in pbar:
        x_lower, x_upper, y_batch = next(iter(train_loader))
        x_lower = __flatten(x_lower)
        x_upper = __flatten(x_upper)
        x_batch = Interval(lower=x_lower, upper=x_upper)
        y_batch = y_batch.squeeze()

        prediction = model.forward(x_batch)
        loss_context = Context()
        loss_function.forward(prediction, y_batch, loss_context)

        seen_samples = y_batch.size(0)
        correct_predictions = correct_classifications(prediction, y_batch)

        loss_grad = loss_function.backwards(loss_context)
        _, grad_layers = model.backwards(loss_grad)

        model.update_weights(grad_layers, learning_rate)

        train_accuracy = correct_predictions / seen_samples

        validation_accuracy, certified_robustness, certified_accuracy = \
            evaluate_model(model=model, data_loader=val_loader)

        weight_difference = model.layers[0].weights - optimal_weights
        bias_difference = model.layers[0].biases - optimal_bias
        gradients = grad_layers[0]
        complete_weight_term = -2.0 * weight_difference * gradients[0]
        complete_bias_term = -2.0 * bias_difference * gradients[1]

        max_certified_accuracy = max(max_certified_accuracy, certified_accuracy)

        print(
            f"Train accuracy: {train_accuracy:.1%}, "
            f"Validation accuracy: {validation_accuracy:.1%}, "
            f"certified robustness: {certified_robustness:.1%}, "
            f"certified accuracy: {certified_accuracy:.1%} "
            f"weight term: {weight_difference}, "
            f"bias term: {bias_difference}, "
            f"gradients: {gradients}, "
        )
    return max_certified_accuracy


@typechecked
def train_model(
        model: Model,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        loss_function: Objective,
        quantize_inputs: bool = False,
        early_stopping: bool = False
) -> Tuple[float, Model]:
    pbar = trange(epochs)
    max_certified_accuracy = 0.0
    best_model = None
    for _ in pbar:
        seen_samples = 0
        correct_predictions = 0
        num_nonempty_intervals = 0
        for batch in train_loader:
            if len(batch) == 2:
                x_batch, y_batch = batch
                x_batch = __flatten(x_batch)
                x_batch = Interval.empty_interval(x_batch)
            else:
                x_lower, x_upper, y_batch = batch
                x_lower = __flatten(x_lower)
                x_upper = __flatten(x_upper)
                x_batch = Interval(lower=x_lower, upper=x_upper)
            y_batch = y_batch.squeeze()

            if quantize_inputs:
                x_batch = quantize(x_batch)
                num_nonempty_intervals += torch.count_nonzero(x_batch.lower < x_batch.upper)

            prediction = model.forward(x_batch)

            loss_context = Context()
            loss_function.forward(prediction, y_batch, loss_context)

            seen_samples += y_batch.size(0)
            correct_predictions += correct_classifications(prediction, y_batch)

            loss_grad = loss_function.backwards(loss_context)
            _, grad_layers = model.backwards(loss_grad)

            model.update_weights(grad_layers, learning_rate)

        train_accuracy = correct_predictions / seen_samples

        validation_accuracy, certified_robustness, certified_accuracy = \
            evaluate_model(model=model, data_loader=val_loader)
        if certified_accuracy > max_certified_accuracy:
            max_certified_accuracy = certified_accuracy
            best_model = model.clone()
        if early_stopping and certified_accuracy < max_certified_accuracy - 0.05:
            break

        pbar.set_description(
            f"Train accuracy: {train_accuracy:.1%},"
            f"Validation accuracy: {validation_accuracy:.1%},"
            f"certified robustness: {certified_robustness:.1%},"
            f"certified accuracy: {certified_accuracy:.1%}"
        )
    return max_certified_accuracy, best_model


@typechecked
def train_model_timing(
        model: Model,
        pytorch_model: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        learning_rate: float,
        loss_function: Objective,
        pytorch_loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    start_time_bounds = time.time()
    with torch.no_grad():
        for _ in range(epochs):
            for batch in train_loader:
                if len(batch) == 2:
                    x_batch, y_batch = batch
                    x_batch = __flatten(x_batch)
                    x_batch = Interval.empty_interval(x_batch)
                else:
                    x_lower, x_upper, y_batch = batch
                    x_lower = __flatten(x_lower)
                    x_upper = __flatten(x_upper)
                    x_batch = Interval(lower=x_lower, upper=x_upper)
                y_batch = y_batch.squeeze()

                prediction = model.forward(x_batch)

                loss_context = Context()
                loss_function.forward(prediction, y_batch, loss_context)

                loss_grad = loss_function.backwards(loss_context)
                _, grad_layers = model.backwards(loss_grad)

                model.update_weights(grad_layers, learning_rate)
    end_time_bounds = time.time()
    bounds_time = end_time_bounds - start_time_bounds

    start_time_pytorch = time.time()
    for _ in range(epochs):
        for batch in train_loader:
            if len(batch) == 2:
                x_batch, y_batch = batch
                x_batch = __flatten(x_batch)
            else:
                x_lower, x_upper, y_batch = batch
                x_lower = __flatten(x_lower)
                x_upper = __flatten(x_upper)
                x_batch = Interval(lower=x_lower, upper=x_upper).center()
            y_batch = y_batch.squeeze()
            pytorch_model.zero_grad()
            prediction = pytorch_model(x_batch).squeeze()
            loss = pytorch_loss_function(prediction, y_batch.float())
            loss.backward()
            optimizer.step()

    end_time_pytorch = time.time()
    pytorch_time = end_time_pytorch - start_time_pytorch

    return bounds_time, pytorch_time


@typechecked
def quantize(inputs: Interval) -> Interval:
    require(torch.all(inputs.lower >= 0.0) and torch.all(inputs.upper <= 1.0), "Expected values in range 0..1")
    lower_bounds, upper_bounds = inputs.lower, inputs.upper
    lower_bounds[lower_bounds < .5] = 0.0
    lower_bounds[lower_bounds >= .5] = 1.0
    upper_bounds[upper_bounds < .5] = 0.0
    upper_bounds[upper_bounds >= .5] = 1.0
    return Interval(lower=lower_bounds, upper=upper_bounds)


@typechecked
def evaluate_model(
        model: Model,
        data_loader: DataLoader
) -> Tuple[float, float, float]:
    seen_samples = 0
    correct_predictions = 0
    robust_predictions = 0
    robust_correct_predictions = 0

    for batch in data_loader:
        if len(batch) == 2:
            x_batch, y_batch = batch
            x_batch = __flatten(x_batch)
            x_batch = Interval.empty_interval(x_batch)
        else:
            x_lower, x_upper, y_batch = batch
            x_lower = __flatten(x_lower)
            x_upper = __flatten(x_upper)
            x_batch = Interval(lower=x_lower, upper=x_upper)
        y_batch = y_batch.squeeze()

        prediction = model.forward(x_batch)

        seen_samples += y_batch.size(0)
        certified_robust, certified_correct = certified_classifications(prediction, y_batch)
        correct_predictions += correct_classifications(prediction, y_batch)
        robust_predictions += certified_robust
        robust_correct_predictions += certified_correct

    accuracy = correct_predictions / seen_samples
    certified_robustness = robust_predictions / seen_samples
    certified_accuracy = robust_correct_predictions / seen_samples
    return accuracy, certified_robustness, certified_accuracy


@typechecked
def evaluate_ensemble(
        models: List[Model],
        data_loader: DataLoader
) -> Tuple[float, float]:
    seen_samples = 0
    robust_predictions = 0
    robust_correct_predictions = 0

    for batch in data_loader:
        if len(batch) == 2:
            x_batch, y_batch = batch
            x_batch = __flatten(x_batch)
            x_batch = Interval.empty_interval(x_batch)
        else:
            x_lower, x_upper, y_batch = batch
            x_lower = __flatten(x_lower)
            x_upper = __flatten(x_upper)
            x_batch = Interval(lower=x_lower, upper=x_upper)
        y_batch = y_batch.squeeze()

        predictions = [model.forward(x_batch) for model in models]
        seen_samples += y_batch.size(0)
        certified_robust, certified_correct = certified_classifications_ensemble(predictions, y_batch)
        robust_predictions += certified_robust
        robust_correct_predictions += certified_correct

    certified_robustness = robust_predictions / seen_samples
    certified_accuracy = robust_correct_predictions / seen_samples
    return certified_robustness, certified_accuracy


@typechecked
def __flatten(x: Tensor) -> Tensor:
    return torch.reshape(x, (x.size(0), -1))
