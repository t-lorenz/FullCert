import unittest

import torch

import boundflow.interval as iv
from assertions import almost_contains, uniform_from, uniform_interval, uniform, uniform_target
from boundflow.context import Context
from boundflow.objective import BinaryCrossEntropyLoss, Objective

BATCH_SIZE = 1000
NUM_CLASSES = 2


class TestBCELoss(unittest.TestCase):

    def test_bce_tensor(self):
        objective = BinaryCrossEntropyLoss()
        torch_objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.__test_loss_tensor(objective, torch_objective)

    def test_bce_empty_interval(self):
        iv_objective = BinaryCrossEntropyLoss()
        torch_objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.__test_loss_empty_interval(iv_objective, torch_objective)

    def test_bce_intervals(self):
        iv_objective = BinaryCrossEntropyLoss()
        torch_objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        interval_size = 1000
        self.__test_loss_interval(iv_objective, torch_objective, interval_size)

    def test_bce_tiny_intervals(self):
        iv_objective = BinaryCrossEntropyLoss()
        torch_objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        interval_size = 1e-7
        self.__test_loss_interval(iv_objective, torch_objective, interval_size)

    @staticmethod
    def __test_loss_empty_interval(iv_objective: Objective, torch_objective: torch.nn.Module):
        context = Context()

        x = uniform(size=(BATCH_SIZE, 1), lower=-1000, upper=1000)
        logits = iv.Interval.empty_interval(x)
        targets = uniform_target(size=BATCH_SIZE, num_classes=NUM_CLASSES)

        actual_loss = iv_objective.forward(logits, targets, context)
        actual_gradient = iv_objective.backwards(context)

        x.requires_grad_()
        expected_loss = torch_objective(x.squeeze(), targets.float())
        expected_loss.backward()
        expected_gradient = x.grad
        torch.testing.assert_close(actual_loss.lower, expected_loss)
        torch.testing.assert_close(actual_loss.upper, expected_loss)
        torch.testing.assert_close(actual_gradient.lower, expected_gradient)
        torch.testing.assert_close(actual_gradient.upper, expected_gradient)

    @staticmethod
    def __test_loss_tensor(iv_objective: Objective, torch_objective: torch.nn.Module):
        context = Context()

        logits = uniform(size=(BATCH_SIZE, 1), lower=-1000, upper=1000)
        targets = uniform_target(size=BATCH_SIZE, num_classes=NUM_CLASSES)

        actual_loss = iv_objective.forward(logits, targets, context)
        actual_gradient = iv_objective.backwards(context)

        logits.requires_grad_()
        expected_loss = torch_objective(logits.squeeze(), targets.float())
        expected_loss.backward()
        expected_gradient = logits.grad
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def __test_loss_interval(self, iv_objective: Objective, torch_objective: torch.nn.Module, interval_size: float, ):
        context = Context()

        logits = uniform_interval(size=(BATCH_SIZE, 1), lower=-interval_size, upper=interval_size)
        targets = uniform_target(size=BATCH_SIZE, num_classes=NUM_CLASSES)
        actual_loss = iv_objective.forward(logits, targets, context)
        actual_gradients = iv_objective.backwards(context)

        for _ in range(1000):
            samples = uniform_from(logits)
            samples.requires_grad_()
            expected_loss = torch_objective(samples.squeeze(), targets.float())
            self.assertTrue(almost_contains(actual_loss, expected_loss))
            expected_loss.backward()
            expected_gradients = samples.grad
            self.assertTrue(almost_contains(actual_gradients, expected_gradients))
