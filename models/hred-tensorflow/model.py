import numpy as np
import tensorflow as tf
import time

from tensorflow.python.ops.nn import dynamic_rnn
from utils.output_projection import output_projection_layer, MyDense
from utils import SummaryHelper

class HredModel(object):
	def __init__(self, data, args, embed):
		self.init_states = tf.placeholder(tf.float32, (None, args.ch_size), 'ctx_inps')  # batch*ch_size
		self.posts = tf.placeholder(tf.int32, (None, None), 'enc_inps')  # batch*len
		self.posts_length = tf.placeholder(tf.int32, (None,), 'enc_lens')  # batch
		self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
		self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch

		# deal with original data to adapt encoder and decoder
		batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
		self.responses = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1] # no go_id
		self.responses_length = self.origin_responses_length - 1
		self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0] # no eos_id
		self.responses_target = self.responses
		decoder_len = decoder_len - 1
		self.posts_input = self.posts   # batch*len
		self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
			decoder_len), reverse=True, axis=1), [-1, decoder_len])

		# initialize the training process
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		# build the embedding table and embedding input
		if embed is None:
			# initialize the embedding randomly
			self.embed = tf.get_variable('embed', [data.vocab_size, args.embedding_size], tf.float32)
		else:
			# initialize the embedding by pre-trained word vectors
			self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

		self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_input) #batch*len*unit
		self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)

		# build rnn_cell
		cell_enc = tf.nn.rnn_cell.GRUCell(args.eh_size)
		cell_ctx = tf.nn.rnn_cell.GRUCell(args.ch_size)
		cell_dec = tf.nn.rnn_cell.GRUCell(args.dh_size)

		# build encoder
		with tf.variable_scope('encoder'):
			encoder_output, encoder_state = dynamic_rnn(cell_enc, self.encoder_input,
				self.posts_length, dtype=tf.float32, scope="encoder_rnn")

		with tf.variable_scope('context'):
			_, self.context_state = cell_ctx(encoder_state, self.init_states)

		# get output projection function
		output_fn = MyDense(data.vocab_size, use_bias = True)
		sampled_sequence_loss = output_projection_layer(args.dh_size, data.vocab_size, args.softmax_samples)

		# construct helper and attention
		train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, tf.maximum(self.responses_length, 1))
		infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id)
		attn_mechanism = tf.contrib.seq2seq.LuongAttention(args.dh_size, encoder_output,
				memory_sequence_length=tf.maximum(self.posts_length, 1))
		cell_dec_attn = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attn_mechanism,
				attention_layer_size=args.dh_size)
		ctx_state_shaping = tf.layers.dense(self.context_state, args.dh_size, activation = None)
		dec_start = cell_dec_attn.zero_state(batch_size, dtype = tf.float32).clone(cell_state = ctx_state_shaping)

		# build decoder (train)
		with tf.variable_scope('decoder'):
			decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, train_helper, dec_start)
			train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished = True, scope = "decoder_rnn")
			self.decoder_output = train_outputs.rnn_output
			self.decoder_distribution_teacher, self.decoder_loss = sampled_sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)

		# build decoder (test)
		with tf.variable_scope('decoder', reuse=True):
			decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, infer_helper, dec_start, output_layer = output_fn)
			infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished = True,
					maximum_iterations=args.max_sent_length, scope = "decoder_rnn")
			self.decoder_distribution = infer_outputs.rnn_output
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
				[2, data.vocab_size-2], 2)[1], 2) + 2 # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)
		gradients = tf.gradients(self.decoder_loss, self.params)
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

		self.trainSummary = self.summaryHelper.addGroup(scalar=["loss", "perplexity"], prefix="train")

		scalarlist = ["loss", "perplexity"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
				embedding=emblist, prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
				embedding=emblist, prefix="test")


	def print_parameters(self):
		for item in self.params:
			print('%s: %s' % (item.name, item.get_shape()))

	def step_decoder(self, sess, data, forward_only=False, inference=False):
		input_feed = {
				self.init_states: data['init_states'],
				self.posts: data['posts'],
				self.posts_length: data['posts_length'],
				self.origin_responses: data['responses'],
				self.origin_responses_length: data['responses_length'],
				}

		if inference:
			output_feed = [self.generation_index, self.context_state]
		else:
			if forward_only:
				output_feed = [self.decoder_loss, self.decoder_distribution_teacher, self.context_state]
			else:
				output_feed = [self.decoder_loss, self.gradient_norm, self.update, self.context_state]

		return sess.run(output_feed, input_feed)

	def get_step_data(self, step_data, batched_data, turn):
		current_batch_size = batched_data['sent'].shape[0]
		max_turn_length = batched_data['sent'].shape[1]
		max_sent_length = batched_data['sent'].shape[2]
		if turn == -1:
			step_data['posts'] = np.zeros((current_batch_size, 1), dtype=int)
		else:
			step_data['posts'] = batched_data['sent'][:, turn, :]
		step_data['responses'] = batched_data['sent'][:, turn + 1, :]
		step_data['posts_length'] = np.zeros((current_batch_size,), dtype=int)
		step_data['responses_length'] = np.zeros((current_batch_size,), dtype=int)
		for i in range(current_batch_size):
			if turn < len(batched_data['sent_length'][i]):
				if turn == -1:
					step_data['posts_length'][i] = 1
				else:
					step_data['posts_length'][i] = batched_data['sent_length'][i][turn]
			if turn + 1 < len(batched_data['sent_length'][i]):
				step_data['responses_length'][i] = batched_data['sent_length'][i][turn + 1]
		max_posts_length = np.max(step_data['posts_length'])
		max_responses_length = np.max(step_data['responses_length'])
		step_data['posts'] = step_data['posts'][:, 0:max_posts_length]
		step_data['responses'] = step_data['responses'][:, 0:max_responses_length]

	def train_step(self, sess, data, args):
		current_batch_size = data['sent'].shape[0]
		max_turn_length = data['sent'].shape[1]
		max_sent_length = data['sent'].shape[2]
		loss = np.zeros((1,))
		total_length = np.zeros((1,))
		step_data = {}
		context_states = np.zeros((current_batch_size, args.ch_size))

		for turn in range(max_turn_length - 1):
			self.get_step_data(step_data, data, turn)
			step_data['init_states'] = context_states
			decoder_loss, _, _, context_states = self.step_decoder(sess, step_data)
			length = np.sum(np.maximum(step_data['responses_length'] - 1, 0))
			total_length += length
			loss += decoder_loss * length
		return loss / total_length


	def evaluate(self, sess, data, key_name, args):
		loss = np.zeros((1,))
		total_length = np.zeros((1,))
		data.restart(key_name, batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			current_batch_size = batched_data['sent'].shape[0]
			max_turn_length = batched_data['sent'].shape[1]
			max_sent_length = batched_data['sent'].shape[2]
			step_data = {}
			context_states = np.zeros((current_batch_size, args.ch_size))
			for turn in range(max_turn_length - 1):
				self.get_step_data(step_data, batched_data, turn)
				step_data['init_states'] = context_states
				decoder_loss, _, context_states = self.step_decoder(sess, step_data, forward_only=True)
				length = np.sum(np.maximum(step_data['responses_length'] - 1, 0))
				total_length += length
				loss += decoder_loss * length
			batched_data = data.get_next_batch(key_name)
		loss /= total_length

		print('	perplexity on %s set: %.2f' % (key_name, np.exp(loss)))
		return loss

	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((1,)), .0, 0
		previous_losses = [1e18] * 5
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")
		for epoch_step in range(args.epochs):
			while batched_data != None:
				step_cnt = 1
				for step in range(args.checkpoint_steps):
					if batched_data == None:
						break
					start_time = time.time()
					loss_step += self.train_step(sess, batched_data, args)
					time_step += time.time() - start_time
					batched_data = data.get_next_batch("train")
					step_cnt += 1

				loss_step /= step_cnt
				time_step /= step_cnt
				show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
				print("Epoch %d global step %d learning rate %.4f step-time %.2f perplexity %s"
					  % (epoch_step, self.global_step.eval(), self.learning_rate.eval(),
						 time_step, show(np.exp(loss_step))))
				self.trainSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': loss_step, 'perplexity': np.exp(loss_step)})
				#self.saver.save(sess, '%s/checkpoint_latest' % args.model_dir, global_step=self.global_step)\
				self.store_checkpoint(sess, '%s/checkpoint_latest/checkpoint' % args.model_dir, "latest")

				dev_loss = self.evaluate(sess, data, "dev", args)
				self.devSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': dev_loss, 'perplexity': np.exp(dev_loss)})

				test_loss = self.evaluate(sess, data, "test", args)
				self.testSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': test_loss, 'perplexity': np.exp(test_loss)})

				if np.sum(loss_step) > max(previous_losses):
					sess.run(self.learning_rate_decay_op)
				if dev_loss < best_valid:
					best_valid = dev_loss
					self.store_checkpoint(sess, '%s/checkpoint_best/checkpoint' % args.model_dir, "best")

				previous_losses = previous_losses[1:] + [np.sum(loss_step)]
				loss_step, time_step = np.zeros((1,)), .0


			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")

	def test_process(self, sess, data, args):
		def get_batch_results(batched_responses_id, data):
			batch_results = []
			for response_id in batched_responses_id:
				response_id_list = response_id.tolist()
				response_token = data.convert_ids_to_tokens(response_id_list)
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id)+1]
				else:
					result_id = response_id_list
				batch_results.append(result_id)
			return batch_results

		def padding(matrix):
			l = max([len(d) for d in matrix])
			res = [d + [data.pad_id] * (l - len(d)) for d in matrix]
			return res

		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")
		cnt = 0
		start_time = time.time()
		while batched_data != None:
			current_batch_size = batched_data['sent'].shape[0]
			max_turn_length = batched_data['sent'].shape[1]
			max_sent_length = batched_data['sent'].shape[2]
			if cnt > 0 and cnt % 10 == 0:
				print('processing %d batch data, time cost %.2f s/batch' % (cnt, (time.time() - start_time) / 10))
				start_time = time.time()
			cnt += 1
			step_data = {}
			context_states = np.zeros((current_batch_size, args.ch_size))
			batched_gen_prob = []
			batched_gen = []
			for turn in range(max_turn_length):
				self.get_step_data(step_data, batched_data, turn - 1)
				step_data['init_states'] = context_states
				decoder_loss, gen_prob, context_states = self.step_decoder(sess, step_data, forward_only=True)
				batched_responses_id, context_states = self.step_decoder(sess, step_data, inference=True)
				batch_results = get_batch_results(batched_responses_id, data)
				step_data['gen_prob'] = gen_prob
				batched_gen_prob.append(step_data['gen_prob'])
				step_data['generations'] = batch_results
				batched_gen.append(step_data['generations'])

			def transpose(batched_gen_prob):
				batched_gen_prob_temp = [[0 for i in range(max_turn_length)] for j in range(current_batch_size)]
				for i in range(max_turn_length):
					for j in range(current_batch_size):
						batched_gen_prob_temp[j][i] = batched_gen_prob[i][j]
				batched_gen_prob[:] = batched_gen_prob_temp[:]
				for i in range(current_batch_size):
					for j in range(max_turn_length):
						batched_gen_prob[i][j] = np.concatenate((batched_gen_prob[i][j], [batched_gen_prob[i][j][-1]]), axis=0)
			transpose(batched_gen_prob)
			transpose(batched_gen)

			sent_length = []
			for i in range(current_batch_size):
				sent_length.append(np.array(batched_data['sent_length'][i])+1)
			batched_sent = np.zeros((current_batch_size, max_turn_length, max_sent_length + 2), dtype=int)
			empty_sent = np.zeros((current_batch_size, 1, max_sent_length + 2), dtype=int)
			for i in range(current_batch_size):
				for j, _ in enumerate(sent_length[i]):
					batched_sent[i][j][0] = data.go_id
					batched_sent[i][j][1:sent_length[i][j]] = batched_data['sent'][i][j][0:sent_length[i][j]-1]
				empty_sent[i][0][0] = data.go_id
				empty_sent[i][0][1] = data.eos_id

			metric1_data = {
					'sent_allvocabs': batched_data['sent_allvocabs'],
					'sent_length': batched_data['sent_length'],
					'multi_turn_gen_log_prob': batched_gen_prob,
					}
			metric1.forward(metric1_data)
			metric2_data = {
					'context_allvocabs': [],
					'sent_allvocabs': batched_data['sent_allvocabs'],
					'turn_length': batched_data['turn_length'],
					'multi_turn_gen': batched_gen,
					}
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
			for i, reference in enumerate(res['reference']):
				f.write("session: \t%d\n" % i)
				for j in range(len(reference)):
					f.write("\tresp:\t%s\n" % " ".join(reference[j]))
					f.write("\tgen:\t%s\n\n" % " ".join(res['gen'][i][j]))
				f.write("\n")
			# for i in range(len(res['context'])):
			# 	f.write("batch number:\t%d\n" % i)
			# 	for j in range(min(len(res['context'][i]), len(res['reference'][i]))):
			# 		if j > 0 and " ".join(res['context'][i][j]) != " ".join(res['reference'][i][j-1]):
			# 			f.write("\n")
			# 		f.write("post:\t%s\n" % " ".join(res['context'][i][j]))
			# 		f.write("resp:\t%s\n" % " ".join(res['reference'][i][j]))
			# 		if j < len(res['gen'][i]):
			# 			f.write("gen:\t%s\n" % " ".join(res['gen'][i][j]))
			# 		else:
			# 			f.write("gen:\n")

		print("result output to %s" % test_file)
		return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}
