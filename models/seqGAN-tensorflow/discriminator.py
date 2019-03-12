import tensorflow as tf
import numpy as np
import time
from utils import SummaryHelper
# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, args, data, embed, summary, sequence_length, latest_dir, best_dir):
        # Placeholders for input, output and dropout
        self.sequence_length = sequence_length
        self.args = args
        self.data = data

        self.num_classes = args.num_classes
        self.vocab_size = data.vocab_size
        self.filter_sizes = args.dis_filter_sizes
        self.num_filters = args.dis_num_filters
        self.embedding_size = args.embedding_size
        self.l2_reg_lambda = args.dis_l2_reg_lambda
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)
        self.latest_dir = latest_dir
        self.best_dir = best_dir
        self.latest_saver = None
        self.best_saver = None
        self.global_step = tf.Variable(0, trainable=False)

        # initialize the training process
        self.learning_rate = tf.Variable(float(args.dis_lr), trainable=False, dtype=tf.float32)
        _, _, _, self.trainSummary, self.devSummary, self.testSummary = summary

    def build_discriminator(self):
        self.W = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
            name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        
        # Combine all the pooled features
        num_filters_total = sum(self.num_filters)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add highway
        with tf.name_scope("highway"):
            self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.scores, 1), tf.int32), tf.cast(tf.argmax(self.input_y, 1), tf.int32)), tf.float32))

        self.params = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, self.args.grad_clip)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params),
				global_step=self.global_step)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, generator, session, data, forward_only=False):
        sentence = []
        for s in data["sent"]:
            sentence.append(np.concatenate((s[1:],[self.data.word2id["<pad>"]] * (self.sequence_length-len(s[1:]))), 0))
        neg_data = generator.generate(session)

        feed = {
            self.input_x: np.concatenate((np.array(sentence), neg_data), 0) ,
            self.input_y: np.concatenate((np.array([[0,1]]*np.shape(sentence)[0]), [[1,0]]*np.shape(neg_data)[0]), 0),
            self.dropout_keep_prob: 1.0 if forward_only else self.args.dis_dropout_keep_prob
        }

        if forward_only:
            output_feed = [self.loss, self.acc]
        else:
            output_feed = [self.loss, self.acc,
                           self.gradient_norm,
                           self.update]
        return session.run(output_feed, feed)


    def evaluate(self, generator, session, data, batch_size, key_name):
        loss_step, acc_step = 0., 0.
        times = 0
        data.restart(key_name, batch_size=batch_size, shuffle=False)
        batched_data = data.get_next_batch(key_name)
        while batched_data != None:
            outputs = self.step_decoder(generator, session, batched_data, forward_only=True)
            loss_step += outputs[0]
            acc_step += outputs[1]
            times += 1
            batched_data = data.get_next_batch(key_name)

        loss_step /= times
        acc_step /= times

        return loss_step, acc_step

    def store_checkpoint(self, sess, path, key, generator):
        if key == "latest":
            self.latest_saver.save(sess, path, global_step = self.global_step + generator.global_step)
        else:
            self.best_saver.save(sess, path, global_step = self.args.global_step + generator.global_step)


    def train_process(self, generator, data, sess, steps):
        loss_step, acc_step, time_step, epoch = 0., 0., 0., 0
        data.restart("train", batch_size=self.args.batch_size, shuffle=True)
        batched_data = data.get_next_batch("train")
        for epoch in range(steps):
            while batched_data != None:
                global_step = self.global_step.eval()
                if global_step % self.args.checkpoint_steps == 0 and global_step != 0:
                    print("Dis epoch %d global step %d learning rate %.4f step-time %.2f loss_step %.4f acc %.4f"
                            % (epoch, global_step, self.learning_rate.eval(), time_step, loss_step, acc_step))
                    self.trainSummary(global_step // self.args.checkpoint_steps, {'loss': loss_step, 'accuracy':acc_step})                    
                    devout = self.evaluate(generator, sess, data, self.args.batch_size, "dev")
                    print('    dev loss: %.4f, acc: %.4f' % (devout[0], devout[1]))
                    self.devSummary(global_step // self.args.checkpoint_steps, {'loss': devout[0], 'accuracy':devout[1]})                    
                    testout = self.evaluate(generator, sess, data, self.args.batch_size, "test")
                    print('    test loss: %.4f, acc: %.4f' % (testout[0], testout[1]))
                    self.testSummary(global_step // self.args.checkpoint_steps, {'loss': testout[0], 'accuracy':testout[1]})
                    loss_step, acc_step, time_step = np.zeros((1,)), .0, .0
                    self.store_checkpoint(sess, self.latest_dir + '/checkpoint', "latest", generator)

                start_time = time.time()
                outputs = self.step_decoder(generator, sess, batched_data)
                loss_step += outputs[0] / self.args.checkpoint_steps
                acc_step += outputs[1] / self.args.checkpoint_steps
                time_step += (time.time() - start_time) / self.args.checkpoint_steps
                batched_data = data.get_next_batch("train")

            data.restart("train", batch_size=self.args.batch_size, shuffle=True)
            batched_data = data.get_next_batch("train")
