import random
import numpy as np


class TensorTools:

    @staticmethod
    def zero_padding(mini_batch, padding):
        mini_batch_padded = np.pad(mini_batch, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)

        return mini_batch_padded

    @staticmethod
    def create_max_mask(input2d):
        mask = input2d == np.max(input2d)
        return mask
