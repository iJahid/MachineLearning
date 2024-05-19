import numpy as np
from TensorTools import *


class TensorMath:

    @staticmethod
    def convolution(mini_batch, filters, bias, stride, padding):
        """
        Implements the forward propagation for a convolution function

        Arguments:
        mini_batch -- Output activations from the previous layer: numpy array of shape (m, in_c, in_h, in_w)
        filters -- Filters: numpy array of shape (num_filters, in_c, filter_size, filter_size)
        b -- Biases, numpy array of shape (num_filters)

        Returns:
        Z -- conv output, numpy array of shape (m, num_filters, out_h, out_w)
        cache -- cache of values needed for the conv_backward() function
        """
        (m, in_c, in_h, in_w) = mini_batch.shape

        (num_filters, in_c, filter_size, filter_size) = filters.shape
        out_h = int((in_h - filter_size + 2 * padding) / stride) + 1
        out_w = int((in_w - filter_size + 2 * padding) / stride) + 1

        output = np.zeros((m, num_filters, out_h, out_w))

        mini_batch_padded = TensorTools.zero_padding(mini_batch, padding)

        for i in range(m):
            example = mini_batch_padded[i]
            for h in range(out_h):
                for w in range(out_w):
                    for c in range(num_filters):
                        h_start = h * stride
                        h_end = h_start + filter_size
                        w_start = w * stride
                        w_end = w_start + filter_size

                        example_slice_3D = example[:, h_start:h_end, w_start:w_end]
                        convolved_slice_2D = np.multiply(example_slice_3D, filters[c, ...]) + bias[c, ...]
                        output[i, c, h, w] = np.sum(convolved_slice_2D)

        assert (output.shape == (m, num_filters, out_h, out_w))

        cache = (mini_batch, filters, bias)

        return output, cache

    @staticmethod
    def backprop_convolution(upstream_gradient, cache, stride, padding):
        (mini_batch, filters, bias) = cache

        (m, in_c, in_h, in_w) = mini_batch.shape

        (num_filters, in_c, filter_size, filter_size) = filters.shape

        (m, num_filters, out_h, out_w) = upstream_gradient.shape

        grad_input = np.zeros((m, in_c, in_h, in_w))
        grad_filters = np.zeros((num_filters, in_c, filter_size, filter_size))
        grad_bias = np.zeros((num_filters, 1, 1, 1))

        input_padded = TensorTools.zero_padding(mini_batch, padding)
        grad_input_padded = TensorTools.zero_padding(grad_input, padding)

        for i in range(m):
            example_padded = input_padded[i]
            example_grad_padded = grad_input_padded[i]

            for h in range(out_h):
                for w in range(out_w):
                    for c in range(num_filters):

                        h_start = h * stride
                        h_end = h_start + filter_size
                        w_start = w * stride
                        w_end = w_start + filter_size

                        example_padded_slice_3D = example_padded[:, h_start:h_end, w_start:w_end]

                        d = upstream_gradient[i, c, h, w]

                        filter_contribution = filters[c, :, :, :] * d
                        example_grad_padded[:, h_start:h_end, w_start:w_end] += filter_contribution
                        grad_filters[c, :, :, :] += example_padded_slice_3D * d
                        grad_bias[c, :, :, :] += d

            grad_input[i, :, :, :] = example_grad_padded[:, padding:-padding, padding:-padding]

        return grad_input, grad_filters, grad_bias


    @staticmethod
    def max_pooling(mini_batch, pool_size, stride):
        (m, in_c, in_h, in_w) = mini_batch.shape

        out_c = in_c
        out_h = int(1 + (in_h - pool_size) / stride)
        out_w = int(1 + (in_w - pool_size) / stride)

        output = np.zeros((m, out_c, out_h, out_w))

        for i in range(m):
            for c in range(out_c):
                for h in range(out_h):
                    for w in range(out_w):

                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size

                        example_slice_2D = mini_batch[i, c, h_start:h_end, w_start:w_end]

                        output[i, c, h, w] = np.max(example_slice_2D)

        cache = (mini_batch, pool_size, stride)

        return output, cache

    @staticmethod
    def backprop_max_pooling(upstream_grad, cache):
        (mini_batch, pool_size, stride) = cache

        m, in_c, in_h, in_w = mini_batch.shape
        m, out_c, out_h, out_w = upstream_grad.shape

        input_grad = np.zeros(mini_batch.shape)

        for i in range(m):
            for c in range(out_c):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h
                        h_end = h_start + pool_size
                        w_start = w
                        w_end = w_start + pool_size

                        example_slice_2d = mini_batch[i, c, h_start:h_end, w_start:w_end]

                        mask = TensorTools.create_max_mask(example_slice_2d)

                        d = upstream_grad[i, c, h, w]

                        input_grad[i, c, h_start:h_end, w_start:w_end] += np.multiply(mask, d)

        return input_grad
