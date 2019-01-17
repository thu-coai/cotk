import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.layers import base
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.layers import utils

def output_projection_layer(num_units, num_symbols, num_samples=None, name="my_dense"):
    def sampled_sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder_rnn/%s' % name):
            weights = tf.transpose(tf.get_variable("kernel", [num_units, num_symbols]))
            bias = tf.get_variable("bias", [num_symbols])

            local_dis = tf.nn.log_softmax(tf.einsum('aij,kj->aik', outputs, weights) + bias)
            local_labels = tf.reshape(targets, [-1])
            local_masks = tf.reshape(masks, [-1])

            y_prob = tf.reshape(local_dis, [-1, num_symbols])
            labels_onehot = tf.one_hot(local_labels, num_symbols)
            labels_onehot = tf.clip_by_value(labels_onehot, 0.0, 1.0)
            local_loss = tf.reduce_sum(-labels_onehot * y_prob, 1) * local_masks
            
            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            
            return local_dis, loss / total_size
    
    return sampled_sequence_loss

class MyDense(base.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MyDense, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units, ],
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        if len(output_shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
                                                                   [0]])
            # Reshape the output back to the original ndim of the input.
            outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)
