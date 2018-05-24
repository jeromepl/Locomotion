import tensorflow as tf
import math


# Dense Phase-Functionned layer
# TODO currently only supports random_normal initializer for both the kernel and the bias
def densePF(
    inputs,
    phase,
    units,
    activation=None,
    use_bias=True):

    NB_WEIGHT_SETS = 4 # TODO allow changing this (need to use different interpolation function)
    inputs = tf.convert_to_tensor(inputs)
    shape = inputs.get_shape().as_list()

    weight_sets = []
    for _ in range(NB_WEIGHT_SETS):
        weight_sets.append(tf.random_normal([shape[-1].value, units]))

    kernel = cubic_catmull_rom_spline(weight_sets, phase)

    if use_bias:
        bias_sets = []
        for _ in range(NB_WEIGHT_SETS):
            bias_sets.append(tf.random_normal([units,]))

        bias = cubic_catmull_rom_spline(bias_sets, phase)


    # The following was taken and adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/_impl/keras/layers/core.py#L879
    if len(shape) > 2:
        # Broadcasting is required for the inputs.
        outputs = tf.tensordot(inputs, kernel, [[len(shape) - 1], [0]])
        # Reshape the output back to the original ndim of the input.
        if not tf.executing_eagerly():
            output_shape = shape[:-1] + [units]
            outputs.set_shape(output_shape)
    else:
        outputs = tf.mat_mul(inputs, kernel)
    if use_bias:
        outputs = tf.nn.bias_add(outputs, bias)
    if activation is not None:
        return activation(outputs)
    return outputs


def cubic_catmull_rom_spline(weight_sets, phase):
    alpha = weight_sets # alias for readability
    # Make sure the phase is single-valued
    while len(phase.get_shape().as_list()) > 0:
        phase = phase[0]

    w = (4*phase) % 1

    def k(n):
        return (tf.floor(4*phase) + n - 1) % 4
    
    return alpha[k(1)] \
        + w * (1/2*alpha[k(2)] - 1/2*alpha[k(0)]) \
        + w**2 * (alpha[k(0)] - 5/2*alpha[k(1)] + 2*alpha[k(2)] - 1/2*alpha[k(3)]) \
        + w**3 * (3/2*alpha[k(1)] - 3/2*alpha[k(2)] + 1/2*alpha[k(3)] - 1/2*alpha[k(0)])
