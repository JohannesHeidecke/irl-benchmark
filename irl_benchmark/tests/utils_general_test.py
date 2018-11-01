import numpy as np
import torch

from irl_benchmark.utils.general import to_one_hot


def test_one_hot_numpy():
    # test one single vector:
    for max_val in range(1, 10):
        for hot_val in range(max_val):
            encoded = to_one_hot(hot_val, max_val)
            assert isinstance(encoded, np.ndarray)
            assert encoded.shape == (max_val, )
            assert np.sum(encoded) == 1
            assert encoded[hot_val] == 1

    # test two vectors at once:
    for max_val in range(1, 10):
        for first_hot in range(max_val):
            for second_hot in range(max_val):
                encoded = to_one_hot([first_hot, second_hot], max_val)
                assert isinstance(encoded, np.ndarray)
                assert encoded.shape == (2, max_val)
                assert np.sum(encoded) == 2
                assert encoded[0, first_hot] == 1
                assert encoded[1, second_hot] == 1


def test_one_hot_torch():
    # test one single vector:
    for max_val in range(1, 10):
        for hot_val in range(max_val):
            encoded = to_one_hot(hot_val, max_val, torch.zeros)
            assert isinstance(encoded, type(torch.zeros(1)))
            assert encoded.shape == (max_val, )
            assert torch.sum(encoded) == 1
            assert encoded[hot_val] == 1

    # test two vectors at once:
    for max_val in range(1, 10):
        for first_hot in range(max_val):
            for second_hot in range(max_val):
                encoded = to_one_hot([first_hot, second_hot], max_val,
                                     torch.zeros)
                assert isinstance(encoded, type(torch.zeros(1)))
                assert encoded.shape == (2, max_val)
                assert torch.sum(encoded) == 2
                assert encoded[0, first_hot] == 1
                assert encoded[1, second_hot] == 1
