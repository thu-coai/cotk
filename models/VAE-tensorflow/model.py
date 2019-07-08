import numpy as np
import tensorflow as tf
import time

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.layers.core import Dense
from utils import SummaryHelper


class VAEModel(object):
	def __init__(self, data, args, embed):

		with tf.variable_scope("input"):
			with tf.variable_scope("embedding"):
				# build the embedding table and embedding input
				if embed is None:
					# initialize the embedding randomly
					self.embed = tf.get_variable('embed', [data.vocab_size, args.embedding_size], tf.float32)
				else:
					# initialize the embedding by pre-trained word vectors
					self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

			self.sentence = tf.placeholder(tf.int32, (None, None), 'sen_inps')  # batch*len
			self.sentence_length = tf.placeholder(tf.int32, (None,), 'sen_lens')  # batch
			self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

			batch_size, batch_len = tf.shape(self.sentence)[0], tf.shape(self.sentence)[1]
			self.decoder_max_len = batch_len - 1

			self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sentence)  # batch*len*unit
			self.encoder_len = self.sentence_length

			decoder_input = tf.split(self.sentence, [self.decoder_max_len, 1], 1)[0]  # no eos_id
			self.decoder_input = tf.nn.embedding_lookup(self.embed, decoder_input)  # batch*(len-1)*unit
			self.decoder_target = tf.split(self.sentence, [1, self.decoder_max_len], 1)[1]  # no go_id, batch*(len-1)
			self.decoder_len = self.sentence_length - 1
			self.decoder_mask = tf.sequence_mask(self.decoder_len, self.decoder_max_len, dtype=tf.float32)  # batch*(len-1)

		# initialize the training process
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		# build rnn_cell
		cell_enc = tf.nn.rnn_cell.GRUCell(args.eh_size)
		cell_dec = tf.nn.rnn_cell.GRUCell(args.dh_size)

		# build encoder
		with tf.variable_scope('encoder'):
			encoder_output, encoder_state = dynamic_rnn(cell_enc, self.encoder_input,
														self.encoder_len, dtype=tf.float32, scope="encoder_rnn")

		with tf.variable_scope('recognition_net'):
			recog_input = encoder_state
			self.recog_mu = tf.layers.dense(inputs=recog_input, units=args.z_dim, activation=None, name='recog_mu')
			self.recog_logvar = tf.layers.dense(inputs=recog_input, units=args.z_dim, activation=None, name='recog_logvar')

			epsilon = tf.random_normal(tf.shape(self.recog_logvar), name="epsilon")
			std = tf.exp(0.5 * self.recog_logvar)
			self.recog_z = tf.add(self.recog_mu, tf.multiply(std, epsilon), name='recog_z')

			self.kld = tf.reduce_mean(
				0.5 * tf.reduce_sum(tf.exp(self.recog_logvar) + self.recog_mu * self.recog_mu - self.recog_logvar - 1,
									axis=-1))
			self.prior_z = tf.random_normal(tf.shape(self.recog_logvar), name="prior_z")
			latent_sample = tf.cond(self.use_prior,
									lambda: self.prior_z,
									lambda: self.recog_z,
									name='latent_sample')
			dec_init_state = tf.layers.dense(inputs=latent_sample, units=args.dh_size, activation=None)

		with tf.variable_scope("output_layer", initializer=tf.orthogonal_initializer()):
			self.output_layer = Dense(data.vocab_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
									  use_bias=True)

		with tf.variable_scope("decode", initializer=tf.orthogonal_initializer()):
			train_helper = tf.contrib.seq2seq.TrainingHelper(
				inputs=self.decoder_input,
				sequence_length=self.decoder_len
			)
			train_decoder = tf.contrib.seq2seq.BasicDecoder(
				cell=cell_dec,
				helper=train_helper,
				initial_state=dec_init_state,
				output_layer=self.output_layer
			)
			train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
				decoder=train_decoder,
				maximum_iterations=self.decoder_max_len,
				impute_finished=True
			)
			logits = train_output.rnn_output

			crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=self.decoder_target, logits=logits)
			crossent = tf.reduce_sum(crossent * self.decoder_mask)
			self.sen_loss = crossent / tf.to_float(batch_size)
			self.ppl_loss = crossent / tf.reduce_sum(self.decoder_mask)

			self.decoder_distribution_teacher = tf.nn.log_softmax(logits)

		with tf.variable_scope("decode", reuse=True):
			infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.go_id),
																	data.eos_id)
			infer_decoder = tf.contrib.seq2seq.BasicDecoder(
				cell=cell_dec,
				helper=infer_helper,
				initial_state=dec_init_state,
				output_layer=self.output_layer
			)
			infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
				decoder=infer_decoder,
				maximum_iterations=self.decoder_max_len,
				impute_finished=True
			)
			self.decoder_distribution = infer_output.rnn_output
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
													   [2, data.vocab_size - 2], 2)[1], 2) + 2  # for removing UNK

		self.kl_weights = tf.minimum(tf.to_float(self.global_step) / args.full_kl_step, 1.0)
		self.kl_loss = self.kl_weights * tf.maximum(self.kld, args.min_kl)
		self.loss = self.sen_loss + self.kl_loss

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=args.momentum)
		gradients = tf.gradients(self.loss, self.params)
		clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
				args.grad_clip)
		self.update = opt.apply_gradients(zip(clipped_gradients, self.params),
				global_step=self.global_step)

		# save checkpoint
		self.latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=args.checkpoint_max_to_keep, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
		self.best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

		# create summary for tensorboard
		self.create_summary(args)

	def store_checkpoint(self, sess, path, key):
		if key == "latest":
			self.latest_saver.save(sess, path, global_step = self.global_step)
		else:
			self.best_saver.save(sess, path, global_step = self.global_step)
			#self.best_global_step = self.global_step

	def create_summary(self, args):
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

		self.trainSummary = self.summaryHelper.addGroup(scalar=["loss",
																"perplexity",
																"kl_loss",
																"kld",
																"kl_weight"],
														prefix="train")

		scalarlist = ["loss", "perplexity", "kl_loss", "kld", "kl_weight"]
		tensorlist = []
		textlist = []
		for i in args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
													   prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
													   prefix="test")

	def print_parameters(self):
		for item in self.params:
			print('%s: %s' % (item.name, item.get_shape()))

	def step_decoder(self, session, data, forward_only=False):
		input_feed = {self.sentence: data['sent'],
					  self.sentence_length: data['sent_length'],
					  self.use_prior: False}
		if forward_only:
			output_feed = [self.loss,
						   self.decoder_distribution_teacher,
						   self.ppl_loss,
						   self.kl_loss,
						   self.kld,
						   self.kl_weights]
		else:
			output_feed = [self.loss,
						   self.gradient_norm,
						   self.update,
						   self.ppl_loss,
						   self.kl_loss,
						   self.kld,
						   self.kl_weights
						   ]
		return session.run(output_feed, input_feed)

	def inference(self, session, data):
		input_feed = {self.sentence: data['sent'],
					  self.sentence_length: data['sent_length'],
					  self.use_prior: True}
		output_feed = [self.generation_index]
		return session.run(output_feed, input_feed)

	def evaluate(self, sess, data, batch_size, key_name):
		loss_step = np.zeros((1,))
		ppl_loss_step, kl_loss_step, kld_step, kl_weight_step = 0, 0, 0, 0
		times = 0
		data.restart(key_name, batch_size=batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			outputs = self.step_decoder(sess, batched_data, forward_only=True)
			loss_step += outputs[0]
			ppl_loss_step += outputs[-4]
			kl_loss_step += outputs[-3]
			kld_step += outputs[-2]
			kl_weight_step = outputs[-1]
			times += 1
			batched_data = data.get_next_batch(key_name)

		loss_step /= times
		ppl_loss_step /= times
		kl_loss_step /= times
		kld_step /= times

		print('    loss: %.2f' % loss_step)
		print('    kl_loss: %.2f' % kl_loss_step)
		print('    perplexity: %.2f' % np.exp(ppl_loss_step))
		print('    kld: %.2f' % kld_step)
		return loss_step, ppl_loss_step, kl_loss_step, kld_step, kl_weight_step

	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((1,)), .0, 0
		ppl_loss_step, kl_loss_step, kld_step, kl_weight_step = 0, 0, 0, 0
		previous_losses = [1e18] * 5
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")
		for epoch_step in range(args.epochs):
			while batched_data != None:
				if self.global_step.eval() % args.checkpoint_steps == 0 and self.global_step.eval() != 0:
					print("Epoch %d global step %d learning rate %.4f step-time %.2f"
						  % (epoch_step, self.global_step.eval(), self.learning_rate.eval(),
							 time_step))
					print('    loss: %.2f' % loss_step)
					print('    kl_loss: %.2f' % kl_loss_step)
					print('    perplexity: %.2f' % np.exp(ppl_loss_step))
					print('    kld: %.2f' % kld_step)
					self.trainSummary(self.global_step.eval() // args.checkpoint_steps,
									  {'loss': loss_step,
									   'perplexity': np.exp(ppl_loss_step),
									   'kl_loss': kl_loss_step,
									   'kld': kld_step,
									   'kl_weight': kl_weight_step})
					#self.saver.save(sess, '%s/checkpoint_latest' % args.model_dir, global_step=self.global_step)\
					self.store_checkpoint(sess, '%s/checkpoint_latest/checkpoint' % args.model_dir, "latest")

					devout = self.evaluate(sess, data, args.batch_size, "dev")
					self.devSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': devout[0],
																					   'perplexity': np.exp(devout[1]),
																					   'kl_loss': devout[2],
																					   'kld': devout[3],
																					   'kl_weight': devout[4]})

					testout = self.evaluate(sess, data, args.batch_size, "test")
					self.testSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': testout[0],
																						'perplexity': np.exp(
																							testout[1]),
																						'kl_loss': testout[2],
																						'kld': testout[3],
																						'kl_weight': testout[4]})

					if np.sum(loss_step) > max(previous_losses):
						sess.run(self.learning_rate_decay_op)
					if devout[0] < best_valid:
						best_valid = devout[0]
						self.store_checkpoint(sess, '%s/checkpoint_best/checkpoint' % args.model_dir, "best")

					previous_losses = previous_losses[1:] + [np.sum(loss_step)]
					loss_step, time_step = np.zeros((1,)), .0
					ppl_loss_step, kl_loss_step, kld_step, kl_weight_step = 0, 0, 0, 0

				start_time = time.time()
				outputs = self.step_decoder(sess, batched_data)
				loss_step += outputs[0] / args.checkpoint_steps
				ppl_loss_step += outputs[-4] / args.checkpoint_steps
				kl_loss_step += outputs[-3] / args.checkpoint_steps
				kld_step += outputs[-2] / args.checkpoint_steps
				kl_weight_step = outputs[-1]

				time_step += (time.time() - start_time) / args.checkpoint_steps
				batched_data = data.get_next_batch("train")

			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")

	def test_process(self, sess, data, args):
		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")
		results = []
		while batched_data != None:
			batched_responses_id = self.inference(sess, batched_data)[0]
			gen_log_prob = self.step_decoder(sess, batched_data, forward_only=True)[1]
			metric1_data = {'sent_allvocabs': np.array(batched_data['sent_allvocabs']),
							'sent_length': np.array(batched_data['sent_length']),
							'gen_log_prob': np.array(gen_log_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				result_token = []
				response_id_list = response_id.tolist()
				response_token = data.convert_ids_to_tokens(response_id_list)
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id)+1]
				else:
					result_id = response_id_list
				for token in response_token:
					if token != data.ext_vocab[data.eos_id]:
						result_token.append(token)
					else:
						break
				results.append(result_token)
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results)}
			metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		res = metric1.close()
		res.update(metric2.close())

		test_file = args.out_dir + "/%s_%s.txt" % (args.name, "test")
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
