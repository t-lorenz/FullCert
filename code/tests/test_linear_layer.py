import unittest

import torch
from torch.testing import assert_close

from boundflow.context import Context
from boundflow.layers import LinearLayer


class TestLinearLayer(unittest.TestCase):

    @staticmethod
    def test_forward():
        layer = LinearLayer(2, 2)
        layer.weights = torch.tensor([[1., 1.], [-1., 0.]]).t()
        layer.biases = torch.tensor([[.5, -.5]])

        x = torch.tensor([[1., 2.]])
        context = Context()
        out = layer.forward(x, context)
        expected = torch.tensor([[3.5, -1.5]])

        assert_close(out, expected)

    @staticmethod
    def test_backward():
        layer = LinearLayer(2, 2)
        layer.weights = torch.tensor([[1., 1.], [-1., 0.]]).t()
        layer.biases = torch.tensor([[.5, -.5]])

        x = torch.tensor([[1., 2.]])
        context = Context()
        layer.forward(x, context)
        outer_grad = torch.tensor([[1., 1.]])

        dx, (dw, db) = layer.backwards(outer_grad, context)

        dx_expected = torch.tensor([[0., 1.]])
        assert_close(dx, dx_expected)

        dw_expected = torch.tensor([[1., 1.], [2., 2.]])
        assert_close(dw, dw_expected)

        db_expected = torch.tensor([[1., 1.]])
        assert_close(db, db_expected)


if __name__ == "__main__":
    unittest.main()
