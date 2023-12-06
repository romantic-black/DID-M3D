import unittest
import torch
from lib.losses.loss_function import get_alpha


class TestLossFunction(unittest.TestCase):

    def test_get_alpha(self):
        B = 3
        K = 50
        heading = torch.randn(B * K, 24)
        alpha = get_alpha(heading)
        print(alpha.shape)
        print(alpha)


if __name__ == '__main__':
    unittest.main()
