import tensorflow as tf
import math

import matplotlib.pyplot as plt


# Dense Phase-Functionned layer
# TODO currently only supports random_normal initializer for both the kernel and the bias
def densePF(
        inputs,
        phase,
        units,
        activation=None,
        use_bias=True):

    # TODO allow changing this (need to use different interpolation function)
    NB_WEIGHT_SETS = 4
    inputs = tf.convert_to_tensor(inputs)
    shape = inputs.get_shape().as_list()

    weight_sets = []
    for _ in range(NB_WEIGHT_SETS):
        weight_sets.append(tf.random_normal([shape[-1], units]))

    kernel = cubic_catmull_rom_spline(weight_sets, phase)

    if use_bias:
        bias_sets = []
        for _ in range(NB_WEIGHT_SETS):
            bias_sets.append(tf.random_normal([units, ]))

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
        outputs = tf.matmul(inputs, kernel)
    if use_bias:
        outputs = tf.nn.bias_add(outputs, bias)
    if activation is not None:
        return activation(outputs)
    return outputs


def cubic_catmull_rom_spline(weight_sets, phase):
    alpha = weight_sets  # alias for readability
    # Make sure the phase is single-valued
    while len(phase.get_shape().as_list()) > 0:
        phase = phase[0]

    w = (4*phase) % 1

    def k(n):
        return (tf.cast(4*phase, tf.int32) + n - 1) % 4

    y0 = tf.gather(alpha, k(0)) # basically alpha[k(0)] but where k(0) is a tensor
    y1 = tf.gather(alpha, k(1))
    y2 = tf.gather(alpha, k(2))
    y3 = tf.gather(alpha, k(3))

    return y1 \
        + w * (1/2*y2 - 1/2*y0) \
        + w**2 * (y0 - 5/2*y1 + 2*y2 - 1/2*y3) \
        + w**3 * (3/2*y1 - 3/2*y2 + 1/2*y3 - 1/2*y0)

# def cubic(y0, y1, y2, y3, mu):
#     return (
#         (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu +
#         (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu +
#         (-0.5*y0+0.5*y2)*mu +
#         (y1))


# Test
if __name__ == '__main__':
    tf.enable_eager_execution()

    # densePF test:
    state = tf.constant([[0, 1, -2, 2]], dtype=tf.float32)
    phase = tf.constant([[0]], dtype=tf.float32)
    output = densePF(state, phase, 6, activation=tf.nn.relu)
    print(output)

    # Cubic Catmull Rom Spline test:
    weight_sets = tf.constant([[0], [1], [-2], [2]], dtype=tf.float32)
    results = []
    phases = []
    for phase in tf.range(0, 1, 0.01):
        phases.append(phase)
        results.append(cubic_catmull_rom_spline(weight_sets, phase))

    plt.plot(phases, results)
    plt.show()
