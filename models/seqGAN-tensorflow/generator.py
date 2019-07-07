import tensorflow as tf
import numpy as np
import time
from utils import SummaryHelper

class Generator(object):
    """Generator of SeqGAN implementation based on https://arxiv.org/abs/1609.05473
        "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient"
        Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu
    """
    def __init__(self, args, data, embed, summary, sequence_length, latest_dir, best_dir):
        """ Basic Set up

        Args:
           num_emb: output vocabulary size
           batch_size: batch size for generator
           emb_dim: LSTM hidden unit dimension
           sequence_length: maximum length of input sequence
           start_token: special token used to represent start of sentence
           initializer: initializer for LSTM kernel and output matrix
        """
        self.num_emb = data.vocab_size
        self.emb_dim = args.embedding_size
        self.hidden_dim = args.eh_size
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.embed = embed
        self.momentum = args.momentum
        self.grad_clip = args.grad_clip
        self.data = data
        self.args = args
        self.sequence_length = sequence_length

        self.latest_dir = latest_dir
        self.best_dir = best_dir

        self.latest_saver = None
        self.best_saver = None

        self.global_step = tf.Variable(0, trainable=False)
        # initialize the training process
        self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
        self.trainSummary, self.devSummary, self.testSummary, _, _, _ = summary
    def build_input(self, name):
        """ Buid input placeholder

        Input:
            name: name of network
        Output:
            self.input_seqs_pre (if name == pretrained)
            self.input_seqs_mask (if name == pretrained, optional mask for masking invalid token)
            self.input_seqs_adv (if name == 'adversarial')
            self.rewards (if name == 'adversarial')
        """
        assert name in ['pretrain', 'adversarial', 'sample']
        if name == 'pretrain':
            self.input_seqs_pre = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_pre")
            self.input_seqs_len = tf.placeholder(tf.int32, [None], name="input_seqs_len")
            self.batch_size = tf.shape(self.input_seqs_pre)[0]
        elif name == 'adversarial':
            self.input_seqs_adv = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_adv")
            self.rewards = tf.placeholder(tf.float32, [None, self.sequence_length], name="reward")
            self.batch_size = tf.shape(self.input_seqs_adv)[0]

    def build_pretrain_network(self):
        """ Buid pretrained network

        Input:
            self.input_seqs_pre
            self.input_seqs_mask
        Output:
            self.pretrained_loss
            self.pretrained_loss_sum (optional)
        """
        self.build_input(name="pretrain")
        self.pretrained_loss = 0.0
        self.input_seqs_mask = tf.reshape(tf.cumsum(tf.one_hot(self.input_seqs_len-1,
            self.sequence_length), reverse=True, axis=1), [-1, self.sequence_length])
        with tf.variable_scope("teller"):
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            #with tf.device("/cpu:0"), tf.variable_scope("embedding"):
            if self.embed is None:
                # initialize the embedding randomly
                word_emb_W = tf.get_variable('embed', [self.num_emb, self.emb_dim], tf.float32, self.initializer)
            else:
                # initialize the embedding by pre-trained word vectors
                word_emb_W = tf.get_variable('embed', dtype=tf.float32, initializer=self.embed)

            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length-1):
                    #with tf.device("/cpu:0"):
                    lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_pre[:, j])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    # calculate loss
                    pretrained_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs_pre[:,j+1], logits=logits)
                    pretrained_loss_t = tf.reduce_sum(tf.multiply(pretrained_loss_t, self.input_seqs_mask[:,j+1]))
                    self.pretrained_loss += pretrained_loss_t
                    word_predict = tf.to_int32(tf.argmax(logits, 1))
            self.pretrained_loss /= tf.reduce_sum(self.input_seqs_mask)
            self.pretrained_loss_sum = tf.summary.scalar("pretrained_loss",self.pretrained_loss)

        self.params = [v for v in tf.trainable_variables() if 'teller' in v.name] #Using name 'teller' here to prevent name collision of target LSTM

        opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
        gradients = tf.gradients(self.pretrained_loss, self.params)
        clipped_gradients, self.gradient_norm_pre = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.update_pre = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)


    def build_adversarial_network(self):
        """ Buid adversarial training network

        Input:
            self.input_seqs_adv
            self.rewards
        Output:
            self.gen_loss_adv
        """
        self.build_input(name="adversarial")
        self.softmax_list_reshape = []
        self.softmax_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            #with tf.device("/cpu:0"), tf.variable_scope("embedding"):
            if self.embed is None:
                # initialize the embedding randomly
                word_emb_W = tf.get_variable('embed', [self.num_emb, self.emb_dim], tf.float32, self.initializer)
            else:
                # initialize the embedding by pre-trained word vectors
                word_emb_W = tf.get_variable('embed', dtype=tf.float32, initializer=self.embed)

            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    tf.get_variable_scope().reuse_variables()
                    #with tf.device("/cpu:0"):
                    lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_adv[:, j])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    softmax = tf.nn.softmax(logits)
                    self.softmax_list.append(softmax)
            self.softmax_list_reshape = tf.transpose(self.softmax_list, perm=[1, 0, 2])
            self.gen_loss_adv = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.input_seqs_adv, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(self.softmax_list_reshape, [-1, self.num_emb]), 1e-20, 1.0)
                    ), 1) * tf.reshape(self.rewards, [-1]))

        self.params = [v for v in tf.trainable_variables() if 'teller' in v.name] #Using name 'teller' here to prevent name collision of target LSTM
        opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum)
        gradients = tf.gradients(self.gen_loss_adv, self.params)
        clipped_gradients, self.gradient_norm_adv = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.update_adv = opt.apply_gradients(zip(clipped_gradients, self.params))

    def build_sample_network(self):
        """ Buid sampling network

        Output:
            self.sample_word_list_reshape
        """
        self.build_input(name="sample")
        self.sample_word_list = []
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            #with tf.device("/cpu:0"), tf.variable_scope("embedding"):
            if self.embed is None:
                # initialize the embedding randomly
                word_emb_W = tf.get_variable('embed', [self.num_emb, self.emb_dim], tf.float32, self.initializer)
            else:
                # initialize the embedding by pre-trained word vectors
                word_emb_W = tf.get_variable('embed', dtype=tf.float32, initializer=self.embed)

            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], "float32", self.initializer)

            self.start_token = tf.constant([self.data.word2id["<go>"]]*self.args.batch_size, dtype=tf.int32)
            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    #with tf.device("/cpu:0"):
                    if j == 0:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        state = lstm1.zero_state([self.args.batch_size], tf.float32)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_word)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())
                    logits = tf.matmul(output, output_W)
                    logprob = tf.log(tf.nn.softmax(logits))
                    sample_word = tf.reshape(tf.to_int32(tf.multinomial(logprob, 1)), shape=[self.args.batch_size])
                    self.sample_word_list.append(sample_word) #sequence_length * batch_size
            self.sample_word_list_reshape = tf.transpose(tf.squeeze(tf.stack(self.sample_word_list)), perm=[1,0]) #batch_size * sequene_length
    def build(self):
        """Create all network for pretraining, adversairal training and sampling"""
        self.build_pretrain_network()
        self.build_adversarial_network()
        self.build_sample_network()
    def generate(self, sess):
        """Helper function for sample generation"""
        return sess.run(self.sample_word_list_reshape)


    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def store_checkpoint(self, sess, path, key, discriminator=None):
        if discriminator != None:
            global_step = self.global_step + discriminator.global_step
        else:
            global_step = self.global_step
        if key == "latest":
            self.latest_saver.save(sess, path, global_step = global_step)
        else:
            self.best_saver.save(sess, path, global_step = global_step)

    def pre_step_decoder(self, session, data, forward_only=False):
        sentence = []
        for s in data["sent"]:
            sentence.append(np.concatenate((s,[self.data.word2id["<pad>"]] * (self.sequence_length-len(s))), 0))
        input_feed = {self.input_seqs_pre: sentence,
                      self.input_seqs_len: data['sent_length']}
        if forward_only:
            output_feed = [self.pretrained_loss]
        else:
            output_feed = [self.pretrained_loss,
                           self.gradient_norm_pre,
                           self.update_pre]
        return session.run(output_feed, input_feed)


    def pre_evaluate(self, sess, data, batch_size, key_name):
        loss_step = np.zeros((1,))
        times = 0
        data.restart(key_name, batch_size=batch_size, shuffle=False)
        batched_data = data.get_next_batch(key_name)
        while batched_data != None:
            outputs = self.pre_step_decoder(sess, batched_data, forward_only=True)
            loss_step += outputs[0]
            times += 1
            batched_data = data.get_next_batch(key_name)

        loss_step /= times

        return loss_step

    def pre_train_process(self, sess, data):
        loss_step, time_step, epoch = np.zeros((1,)), .0, 0
        previous_losses = [1e18] * 5
        best_valid = 1e18

        data.restart("train", batch_size=self.args.batch_size, shuffle=True)
        batched_data = data.get_next_batch("train")
        for epoch in range(self.args.gen_pre_epoch_num):
            while batched_data != None:
                global_step = self.global_step.eval()
                if global_step % self.args.checkpoint_steps == 0 and global_step != 0:
                    print("Pre-train epoch %d global step %d learning rate %.4f step-time %.2f loss %.4f"
                            % (epoch, global_step, self.learning_rate.eval(),
                                time_step, loss_step))
                    self.trainSummary(global_step // self.args.checkpoint_steps, {'loss': loss_step})

                    devout = self.pre_evaluate(sess, data, self.args.batch_size, "dev")
                    print('    Pre-train dev loss: %.4f' % (devout[0]))
                    self.devSummary(global_step // self.args.checkpoint_steps, {'loss': devout[0]})

                    testout = self.pre_evaluate(sess, data, self.args.batch_size, "test")
                    print('    Pre-train test loss: %.4f' % (testout[0]))
                    self.testSummary(global_step// self.args.checkpoint_steps, {'loss': devout[0]})

                    self.store_checkpoint(sess, self.latest_dir + '/checkpoint', "latest")

                    if np.sum(loss_step) > max(previous_losses):
                        sess.run(self.learning_rate_decay_op)
                    if devout[0] < best_valid:
                        best_valid = devout[0]
                        self.store_checkpoint(sess, self.best_dir + '/checkpoint', "best")

                    previous_losses = previous_losses[1:] + [np.sum(loss_step)]
                    loss_step, time_step = np.zeros((1,)), .0
                start_time = time.time()
                outputs = self.pre_step_decoder(sess, batched_data)
                loss_step += outputs[0] / self.args.checkpoint_steps
                time_step += (time.time() - start_time) / self.args.checkpoint_steps
                batched_data = data.get_next_batch("train")

            data.restart("train", batch_size=self.args.batch_size, shuffle=True)
            batched_data = data.get_next_batch("train")

    def test_process(self, sess, data):
        metric = data.get_inference_metric(sample=self.args.sample)
        sample_num = int(len(data.data["test"]["sen"])/self.args.batch_size+1) if self.args.test_sample == None else self.args.test_sample
        for _ in range(sample_num):
            samples = sess.run(self.sample_word_list_reshape)
            metric_data = {'gen': np.array(samples)}
            metric.forward(metric_data)

        res = metric.close()
        test_file = self.args.out_dir + "/%s_%s.txt" % (self.args.name, "test")
        with open(test_file, 'w') as f:
            print("Test Result:")
            for key, value in res.items():
                if isinstance(value, float):
                    print("\t%s:\t%f" % (key, value))
                    f.write("%s:\t%f\n" % (key, value))
            for i in range(len(res['gen'])):
                f.write("%s\n" % " ".join(res['gen'][i]))

        print("result output to %s." % test_file)
        return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}


    def adv_train_process(self, sess, data, rollout_gen, discriminator):
        previous_losses, max_rewards = [1e18] * 5, 0
        for epoch in range(self.args.gen_adv_batch_num):
            def training(tearch_forcing=True):
                if tearch_forcing:
                    data.restart("train", batch_size=self.args.batch_size, shuffle=True)
                    batched_data = data.get_next_batch("train")
                    samples = []
                    for s in batched_data["sent"]:
                        samples.append(np.concatenate((s[1:],[self.data.word2id["<pad>"]] * (self.sequence_length-len(s[1:]))), 0))
                    samples = np.array(samples)
                    length = np.shape(samples)[0]
                    if length < self.args.batch_size:
                        for i in range(self.args.batch_size-length):
                            samples = np.concatenate((samples, np.reshape(samples[i % length], [1, -1])), 0)
                else:
                    samples = sess.run(self.sample_word_list_reshape)
                
                reward_rollout = []
                #calcuate the reward given in the specific stpe t by roll out
                for _ in range(self.args.rollout_num):
                    rollout_list = sess.run(rollout_gen.sample_rollout_step, feed_dict={rollout_gen.pred_seq: samples})
                    rollout_list_stack = np.vstack(rollout_list) #shape: #batch_size * #rollout_step, #sequence length
                    reward_rollout_seq = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.input_x:rollout_list_stack, discriminator.dropout_keep_prob:1.0})
                    reward_last_tok = sess.run(discriminator.ypred_for_auc, feed_dict={discriminator.input_x:samples, discriminator.dropout_keep_prob:1.0})
                    reward_allseq = np.concatenate((reward_rollout_seq, reward_last_tok), axis=0)[:,1]
                    reward_tmp = []
                    for r in range(self.args.batch_size):
                        reward_tmp.append(reward_allseq[range(r, self.args.batch_size * self.sequence_length, self.args.batch_size)])
                    reward_rollout.append(np.array(reward_tmp))
                rewards = np.sum(reward_rollout, axis=0)/self.args.rollout_num
                gen_loss, _, _ = sess.run([self.gen_loss_adv, self.update_adv, self.gradient_norm_adv], feed_dict={self.input_seqs_adv:samples,\
                                                                                            self.rewards:rewards})
                return gen_loss, rewards
            start_time = time.time()
            gen_loss, rewards = training(tearch_forcing=False)
            if gen_loss > max(previous_losses):
                sess.run(self.learning_rate_decay_op)
            if np.mean(rewards) > max_rewards:
                max_rewards = np.mean(rewards)
                self.store_checkpoint(sess, self.best_dir + '/checkpoint', "best", discriminator)

            self.store_checkpoint(sess, self.latest_dir + '/checkpoint', "latest", discriminator)
            previous_losses = previous_losses[1:] + [gen_loss]
            print("Adv train fake batch %d learning rate %.4f step-time %.2f loss %.8f rewards %.8f"
                    % (epoch, self.learning_rate.eval(), time.time()-start_time, gen_loss, np.mean(rewards)))
            self.trainSummary(self.global_step.eval(), {"rewards": np.mean(rewards)})

            if self.args.teacher_forcing:
                start_time = time.time()
                gen_loss, rewards = training(tearch_forcing=True)
                if gen_loss > max(previous_losses):
                    sess.run(self.learning_rate_decay_op)
                previous_losses = previous_losses[1:] + [gen_loss]
                print("Adv train true batch %d learning rate %.4f step-time %.2f loss %.8f rewards %.8f"
                        % (epoch, self.learning_rate.eval(), time.time()-start_time, gen_loss, np.mean(rewards)))
