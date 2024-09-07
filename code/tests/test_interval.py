import unittest

import torch

import boundflow.interval as iv
from assertions import almost_contains, uniform_from, uniform_interval, uniform


class TestIntervalFunctions(unittest.TestCase):

    @staticmethod
    def test_softmax_empty_interval():
        x = uniform(size=(1000, 100), lower=-1000, upper=1000)
        interval = iv.Interval.empty_interval(x)

        result = iv.softmax(interval)

        expected = torch.softmax(x, dim=1)
        torch.testing.assert_close(result.lower, expected)
        torch.testing.assert_close(result.upper, expected)

    def test_softmax(self):
        interval = uniform_interval(size=(1000, 100), lower=-1000, upper=1000)
        result = iv.softmax(interval)

        self.assertTrue(almost_contains(result, torch.softmax(interval.center(), dim=1)))
        self.assertTrue(almost_contains(result, torch.softmax(interval.lower, dim=1)))
        self.assertTrue(almost_contains(result, torch.softmax(interval.upper, dim=1)))
        for _ in range(1000):
            samples = uniform_from(interval)
            self.assertTrue(almost_contains(result, torch.softmax(samples, dim=1)))

    def test_softmax_tiny_intervals(self):
        interval = uniform_interval(size=(1000, 100), lower=-1e-7, upper=1e-7)
        result = iv.softmax(interval)

        self.assertTrue(almost_contains(result, torch.softmax(interval.center(), dim=1)))
        self.assertTrue(almost_contains(result, torch.softmax(interval.lower, dim=1)))
        self.assertTrue(almost_contains(result, torch.softmax(interval.upper, dim=1)))
        for _ in range(1000):
            samples = uniform_from(interval)
            self.assertTrue(almost_contains(result, torch.softmax(samples, dim=1)))

    @staticmethod
    def test_log_softmax_empty_interval():
        x = uniform(size=(1000, 100), lower=-1000, upper=1000)
        interval = iv.Interval.empty_interval(x)

        result = iv.log_softmax(interval)

        expected = torch.log_softmax(x, dim=1)
        torch.testing.assert_close(result.upper, expected)
        torch.testing.assert_close(result.lower, expected)

    def test_log_softmax(self):
        interval = uniform_interval(size=(1000, 100), lower=-1000, upper=1000)
        result = iv.log_softmax(interval)

        self.assertTrue(almost_contains(result, torch.log_softmax(interval.center(), dim=1)))
        self.assertTrue(almost_contains(result, torch.log_softmax(interval.lower, dim=1)))
        self.assertTrue(almost_contains(result, torch.log_softmax(interval.upper, dim=1)))
        for _ in range(1000):
            samples = uniform_from(interval)
            self.assertTrue(almost_contains(result, torch.log_softmax(samples, dim=1)))


if __name__ == "__main__":
    unittest.main()
