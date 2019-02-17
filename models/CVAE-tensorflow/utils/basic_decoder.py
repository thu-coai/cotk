import tensorflow as tf
from tensorflow.contrib.seq2seq import python

class MyBasicDecoder(tf.contrib.seq2seq.BasicDecoder):
    def __init__(self,
                 cell,
                 helper,
                 initial_state,
                 output_layer=None,
                 _aug_context_vector=None
                 ):
        super().__init__(cell, helper, initial_state, output_layer)
        self._aug_context_vector = _aug_context_vector

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.
        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        with tf.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            if self._aug_context_vector is not None:
                inputs = tf.concat((inputs, self._aug_context_vector), 1)
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)

        outputs = python.ops.basic_decoder.BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)
