import tensorflow as tf
class rollout():  
    """Rollout implementation for generator"""
    def __init__(self, args, data, embed, sequence_length):
        #configuraiton setting
        self.sequence_length = sequence_length
        self.hidden_dim = args.eh_size
        self.num_emb = data.vocab_size
        self.emb_dim = args.embedding_size
        self.pred_seq = tf.placeholder(tf.int32, [args.batch_size, self.sequence_length], name="pred_seq_rollout")
        self.batch_size = args.batch_size
        self.sample_rollout_step = []
        self.embed = embed

        #Rollout graph initialization
        with tf.variable_scope("teller"):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
            #with tf.device("/cpu:0"), tf.variable_scope("embedding"):
            if self.embed is None:
                # initialize the embedding randomly
                word_emb_W = tf.get_variable('embed', [self.num_emb, self.emb_dim], tf.float32)
            else:
                # initialize the embedding by pre-trained word vectors
                word_emb_W = tf.get_variable('embed', dtype=tf.float32, initializer=self.embed)

            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.hidden_dim, self.num_emb], tf.float32)

            zero_state = lstm1.zero_state([self.batch_size], tf.float32)
            start_token = tf.constant(data.word2id["<go>"], dtype=tf.int32, shape=[self.batch_size])
            for step in range(1, self.sequence_length):
                #Get the token for i < step
                sample_rollout_left = tf.reshape(self.pred_seq[:, :step], shape=[self.batch_size, step])
                sample_rollout_rihgt = []

                #Update the hidden state for i < step to prepare sampling token for i >= step
                for j in range(step):
                    if j == 0:
                        #with tf.device("/cpu:0"):
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                    else:
                        tf.get_variable_scope().reuse_variables()
                        #with tf.device("/cpu:0"):
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.pred_seq[:, j-1])
                    with tf.variable_scope("lstm"):
                        if j == 0:
                            output, state = lstm1(lstm1_in, zero_state, scope=tf.get_variable_scope())
                        else:
                            output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())
                #Sampling token for i >= step
                for j in range(step, self.sequence_length):
                    #with tf.device("/cpu:0"):
                    if j == step:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.pred_seq[:, j-1])
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_word))
                    with tf.variable_scope("lstm"):
                        output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())
                        logits = tf.matmul(output, output_W)
                        log_probs = tf.log(tf.nn.softmax(logits)+1e-8) #add a tolerance to prevent unmeaningful log value
                        sample_word = tf.to_int32(tf.squeeze(tf.multinomial(log_probs, 1)))
                        sample_rollout_rihgt.append(sample_word)
                sample_rollout_rihgt = tf.transpose(tf.stack(sample_rollout_rihgt))
                sample_rollout = tf.concat([sample_rollout_left, sample_rollout_rihgt], axis=1)
                self.sample_rollout_step.append(sample_rollout)
