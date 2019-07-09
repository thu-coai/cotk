import numpy as np
import tensorflow as tf
import time

from utils import SummaryHelper
from utils.basic_decoder import MyBasicDecoder


class CVAEModel(object):
	def __init__(self, data, args, embed):
		with tf.name_scope("placeholders"):
			self.contexts = tf.placeholder(tf.int32, (None, None, None), 'cxt_inps')  # [batch, utt_len, len]
			self.contexts_length = tf.placeholder(tf.int32, (None,), 'cxt_lens') # [batch]
			self.posts_length = tf.placeholder(tf.int32, (None, None), 'enc_lens')  # [batch, utt_len]
			self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # [batch, len]
			self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # [batch]

		# deal with original data to adapt encoder and decoder
		max_sent_length = tf.shape(self.contexts)[2]
		max_cxt_size = tf.shape(self.contexts)[1]
		self.posts_input = tf.reshape(self.contexts, [-1, max_sent_length]) # [batch * cxt_len, utt_len]
		self.flat_posts_length = tf.reshape(self.posts_length, [-1]) # [batch * cxt_len]

		decoder_len = tf.shape(self.origin_responses)[1]
		self.responses_target = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1] # no go_id
		self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0] # no eoｓ_id
		self.responses_length = self.origin_responses_length - 1
		decoder_len = decoder_len - 1
		self.decoder_mask = tf.sequence_mask(self.responses_length, decoder_len, dtype=tf.float32)
		loss_mask = tf.cast(tf.greater(self.responses_length, 0), dtype=tf.float32)
		batch_size = tf.reduce_sum(loss_mask)

		# initialize the training process
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		with tf.name_scope("embedding"):
			# build the embedding table and embedding input
			if embed is None:
				# initialize the embedding randomly
				self.word_embed = tf.get_variable('word_embed', [data.vocab_size, args.word_embedding_size], tf.float32)
			else:
				# initialize the embedding by pre-trained word vectors
				self.word_embed = tf.get_variable('word_embed', dtype=tf.float32, initializer=embed)
			posts_enc_input = tf.nn.embedding_lookup(self.word_embed, self.posts_input)
			responses_enc_input = tf.nn.embedding_lookup(self.word_embed, self.origin_responses)
			responses_dec_input = tf.nn.embedding_lookup(self.word_embed, self.responses_input)

		with tf.name_scope("cell"):
			# build rnn_cell
			cell_enc_fw = tf.nn.rnn_cell.GRUCell(args.eh_size)
			cell_enc_bw = tf.nn.rnn_cell.GRUCell(args.eh_size)
			cell_ctx = tf.nn.rnn_cell.GRUCell(args.ch_size)
			cell_dec = tf.nn.rnn_cell.GRUCell(args.dh_size)

		# build encoder
		with tf.variable_scope('encoder'):
			posts_enc_output, posts_enc_state = tf.nn.bidirectional_dynamic_rnn(cell_enc_fw,
																				cell_enc_bw,
																				posts_enc_input,
																				self.flat_posts_length,
																				dtype=tf.float32,
																				scope="encoder_bi_rnn")
			posts_enc_state = tf.reshape(tf.concat(posts_enc_state, 1), [-1, max_cxt_size, 2 * args.eh_size])

		with tf.variable_scope('context'):
			_, context_state = tf.nn.dynamic_rnn(cell_ctx,
												 posts_enc_state,
												 self.contexts_length,
												 dtype=tf.float32,
												 scope='context_rnn')
			cond_info = context_state

		with tf.variable_scope("recognition_network"):
			_, responses_enc_state = tf.nn.bidirectional_dynamic_rnn(cell_enc_fw,
																	 cell_enc_bw,
																	 responses_enc_input,
																	 self.responses_length,
																	 dtype=tf.float32,
																	 scope='encoder_bid_rnn')
			responses_enc_state = tf.concat(responses_enc_state, 1)
			recog_input = tf.concat((cond_info, responses_enc_state), 1)
			recog_output = tf.layers.dense(recog_input, 2 * args.latent_size)
			recog_mu, recog_logvar = tf.split(recog_output, 2, 1)
			recog_z = self.sample_gaussian((tf.size(self.contexts_length), args.latent_size), recog_mu, recog_logvar)

		with tf.variable_scope("prior_network"):
			prior_input = cond_info
			prior_fc_1 = tf.layers.dense(prior_input, 2 * args.latent_size, activation=tf.tanh)
			prior_output = tf.layers.dense(prior_fc_1, 2 * args.latent_size)
			prior_mu, prior_logvar = tf.split(prior_output, 2, 1)
			prior_z = self.sample_gaussian((tf.size(self.contexts_length), args.latent_size), prior_mu, prior_logvar)

		with tf.name_scope("decode"):
			# get output projection function
			dec_init_fn = tf.layers.Dense(args.dh_size, use_bias=True)
			output_fn = tf.layers.Dense(data.vocab_size, use_bias=True)

			with tf.name_scope("training"):
				decoder_input = responses_dec_input
				gen_input = tf.concat((cond_info, recog_z), 1)
				dec_init_fn_input = gen_input
				train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, tf.maximum(self.responses_length, 0))
				dec_init_state = dec_init_fn(dec_init_fn_input)
				decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec, train_helper, dec_init_state, output_fn)
				train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished=True,
																		scope="decoder_rnn")
				responses_dec_output = train_outputs.rnn_output
				self.decoder_distribution_teacher = tf.nn.log_softmax(responses_dec_output, 2)

				with tf.name_scope("losses"):
					crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=responses_dec_output,
																			  labels=self.responses_target)
					self.reconstruct_loss = tf.reduce_sum(crossent * self.decoder_mask) / batch_size

					self.KL_loss = tf.reduce_sum(\
						loss_mask * self.KL_divergence(prior_mu, prior_logvar, recog_mu, recog_logvar)) / batch_size
					self.KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / args.full_kl_step)
					self.anneal_KL_loss = self.KL_weight * self.KL_loss

					bow_logits = tf.layers.dense(tf.layers.dense(dec_init_fn_input, 400, activation=tf.tanh),\
												 data.vocab_size)
					tile_bow_logits = tf.tile(tf.expand_dims(bow_logits, 1), [1, decoder_len, 1])
					bow_loss = self.decoder_mask * tf.nn.sparse_softmax_cross_entropy_with_logits(\
						logits=tile_bow_logits,\
						labels=self.responses_target)
					self.bow_loss = tf.reduce_sum(bow_loss) / batch_size

					self.neg_elbo = self.reconstruct_loss + self.KL_loss
					self.train_loss = self.reconstruct_loss + self.anneal_KL_loss + self.bow_loss

			with tf.name_scope("inference"):
				gen_input = tf.concat((cond_info, prior_z), 1)
				dec_init_fn_input = gen_input
				dec_init_state = dec_init_fn(dec_init_fn_input)

				infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.word_embed,
										tf.fill([tf.size(self.contexts_length)], data.go_id),
																		data.eos_id)
				decoder_infer = MyBasicDecoder(cell_dec, infer_helper, dec_init_state, output_layer=output_fn,
											   _aug_context_vector=None)
				infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished=True,
																		maximum_iterations=args.max_sent_length,
																		scope="decoder_rnn")
				self.decoder_distribution = infer_outputs.rnn_output
				self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
														   [2, data.vocab_size - 2], 2)[1], 2) + 2  # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)
		gradients = tf.gradients(self.train_loss, self.params)
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

	def sample_gaussian(self, shape, mu, logvar):
		normal = tf.random_normal(shape=shape, dtype=tf.float32)
		z = tf.exp(logvar / 2) * normal + mu
		return z

	def KL_divergence(self, prior_mu, prior_logvar, recog_mu, recog_logvar):
		KL_divergence = 0.5 * (tf.exp(recog_logvar - prior_logvar) + tf.pow(recog_mu - prior_mu, 2) / tf.exp(
			prior_logvar) - 1 - (recog_logvar - prior_logvar))
		return tf.reduce_sum(KL_divergence, axis=1)

	def store_checkpoint(self, sess, path, key):
		if key == "latest":
			self.latest_saver.save(sess, path, global_step = self.global_step)
		else:
			self.best_saver.save(sess, path, global_step = self.global_step)
			#self.best_global_step = self.global_step

	def create_summary(self, args):
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

		self.trainSummary = self.summaryHelper.addGroup(scalar=["neg_elbo",
																"recontruction_loss",
																"KL_weight",
																"KL_divergence",
																"bow_loss",
																"perplexity"], prefix="train")

		scalarlist = ["neg_elbo", "reconstruction_loss", "KL_weight", "KL_divergence", "bow_loss",
					  "perplexity"]
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


	def _pad_batch(self, raw_batch):
		'''Padding posts_length and trim session, invoked by ^SwitchboardCorpus.split_session^
		and ^SwitchboardCorpus.multi_reference_batches^
		'''
		batch = {'posts_length': [], \
				 'contexts_length': [], \
				 'responses_length': raw_batch['responses_length']}
		max_cxt_size = np.shape(raw_batch['contexts'])[1]
		max_post_len = 0

		for i, speaker in enumerate(raw_batch['posts_length']):
			batch['contexts_length'].append(len(raw_batch['posts_length'][i]))

			if raw_batch['posts_length'][i]:
				max_post_len = max(max_post_len, max(raw_batch['posts_length'][i]))
			batch['posts_length'].append(raw_batch['posts_length'][i] + \
										 [0] * (max_cxt_size - len(raw_batch['posts_length'][i])))

		batch['contexts'] = raw_batch['contexts'][:, :, :max_post_len]
		batch['responses'] = raw_batch['responses'][:, :np.max(raw_batch['responses_length'])]
		return batch

	def _cut_batch_data(self, batch_data, start, end):
		'''Using session[start: end - 1) as context, session[end - 1] as response,
			invoked by ^SwitchboardCorpus.split_session^
		'''
		raw_batch = {'posts_length': [], 'responses_length': []}
		for i in range(len(batch_data['turn_length'])):
			raw_batch['posts_length'].append( \
				batch_data['sent_length'][i][start: end - 1])
			turn_len = len(batch_data['sent_length'][i])
			if end - 1 < turn_len:
				raw_batch['responses_length'].append( \
					batch_data['sent_length'][i][end - 1])
			else:
				raw_batch['responses_length'].append(1)

		raw_batch['contexts'] = batch_data['sent'][:, start: end - 1]
		raw_batch['responses'] = batch_data['sent'][:, end - 1]
		return self._pad_batch(raw_batch)

	def split_session(self, batch_data, session_window, inference=False):
		'''Splits session with different utterances serving as responses

		Arguments:
		    batch_data (dict): must be the same format as the return of self.get_batch
		    inference (bool): True: utterances take turn to serve as responses (without shuffle)
		    				  False: shuffles the order of utterances being responses

		Returns:
		    (list): each element is a dict that contains at least:

				* contexts (:class:`numpy.array`): A 3-d PADDED array containing id of words in contexts.
								Only provide valid words. `unk_id` will be used if a word is not valid.
					Size: `[batch_size, max_turn_length, max_utterance_length]`
				* contexts_length (list): A 1-d list, number of turns
					Size: ^[batch_size]^
				* posts_length (list): A 2-d PADDED list, the length of utterances.
					Size: ^[batch_size, max_turn_length]^
				* responses (:class:`numpy.array`): A 3-d PADDED array containing ids of words in responses.
					Size: ^[batch_size, max_response_length]^
				* responses_length (list): A 1-d list, the length of responses.
					Size: ^[batch_size]^

		'''
		max_turn = np.max(batch_data['turn_length'])
		ends = list(range(2, max_turn + 1))
		if not inference:
			np.random.shuffle(ends)
		for end in ends:
			start = max(end - session_window, 0)
			turn_data = self._cut_batch_data(batch_data, start, end)
			yield turn_data

	def multi_reference_batches(self, data, batch_size):
		'''Get batches of with multiple response candidates

		Arguments:
		     * batch_size (int): batch size

		Returns:
		    (list): each element contains those specified in self.split_session and what follows:

		    	* candidates (list): A 3-d list of response candidates.
		    		Size: [batch_size, _num_candidates, _num_words]

		'''
		data.restart('multi_ref', batch_size, shuffle=False)
		batch_data = data.get_next_batch('multi_ref')
		while batch_data is not None:
			batch = self._cut_batch_data(batch_data,\
							0, np.max(batch_data['turn_length']))
			batch['candidate_allvocabs'] = batch_data['candidate_allvocabs']
			yield batch
			batch_data = data.get_next_batch('multi_ref')


	def step_decoder(self, sess, data, forward_only=False, inference=False):
		input_feed = {
			self.contexts: data['contexts'],
			self.contexts_length: data['contexts_length'],
			self.posts_length: data['posts_length'],
			self.origin_responses: data['responses'],
			self.origin_responses_length: data['responses_length'],
		}
		if inference:
			output_feed = [self.generation_index]
		else:
			if forward_only:
				output_feed = [self.neg_elbo, self.reconstruct_loss, self.KL_weight, self.KL_loss,
							   self.bow_loss, self.decoder_distribution_teacher]
			else:
				output_feed = [self.neg_elbo, self.reconstruct_loss, self.KL_weight, self.KL_loss,
							   self.bow_loss, self.update]

		return sess.run(output_feed, input_feed)

	def train_step(self, sess, data, args):
		output = self.step_decoder(sess, data)
		res = output[:5]
		log_ppl = output[1] * np.sum(np.array(data['responses_length'], dtype=np.int32) > 1) / np.sum(\
			np.maximum(np.array(data['responses_length'], dtype=np.int32) - 1, 0))
		res += [log_ppl]
		return res

	def evaluate(self, sess, data, key_name, args):
		loss_list = np.array([.0] * 6, dtype=np.float32)
		total_length = 0
		total_inst = 0
		data.restart(key_name, batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			for cut_batch_data in self.split_session(batched_data, args.session_window):
				output = self.step_decoder(sess, cut_batch_data, forward_only=True)
				batch_size = np.sum(np.array(cut_batch_data['responses_length'], dtype=np.int32) > 1)
				loss_list[:-1] += np.array(output[:5], dtype=np.float32) * batch_size
				loss_list[-1] += output[1] * batch_size
				total_length += np.sum(np.maximum(np.array(cut_batch_data['responses_length'], dtype=np.int32) - 1, 0))
				total_inst += batch_size
			batched_data = data.get_next_batch(key_name)
		loss_list[:-1] /= total_inst
		loss_list[-1] = np.exp(loss_list[-1] / total_length)

		print('	perplexity on %s set: %.2f' % (key_name, loss_list[-1]))
		return loss_list

	def train_process(self, sess, data, args):
		time_step, epoch_step = .0, 0
		loss_names = ['neg_elbo', 'reconstuction_loss', 'KL_weight', 'KL_divergence', 'bow_loss',
					  'perplexity']
		loss_list = np.array([.0] * len(loss_names), dtype=np.float32)
		previous_losses = [1e18] * 5
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")
		for epoch_step in range(args.epochs):
			while batched_data != None:
				for cut_batch_data in self.split_session(batched_data, args.session_window):
					start_time = time.time()
					output = self.train_step(sess, cut_batch_data, args)
					loss_list += np.array(output[:len(loss_list)], dtype=np.float32)

					time_step += time.time() - start_time

					if (self.global_step.eval() + 1) % args.checkpoint_steps == 0:
						loss_list /= args.checkpoint_steps
						loss_list[-1] = np.exp(loss_list[-1])
						time_step /= args.checkpoint_steps

						print("Epoch %d global step %d learning rate %.4f step-time %.2f perplexity %s"
							  % (epoch_step, self.global_step.eval(), self.learning_rate.eval(),
								 time_step, loss_list[-1]))
						self.trainSummary(self.global_step.eval() // args.checkpoint_steps, \
										  dict(zip(loss_names, loss_list)))
						self.store_checkpoint(sess, '%s/checkpoint_latest/checkpoint' % args.model_dir, "latest")

						dev_loss = self.evaluate(sess, data, "dev", args)
						self.devSummary(self.global_step.eval() // args.checkpoint_steps,\
										dict(zip(loss_names, dev_loss)))

						test_loss = self.evaluate(sess, data, "test", args)
						self.testSummary(self.global_step.eval() // args.checkpoint_steps,\
										 dict(zip(loss_names, test_loss)))

						if loss_list[0] > max(previous_losses):
							sess.run(self.learning_rate_decay_op)
						if dev_loss[0] < best_valid:
							best_valid = dev_loss[0]
							self.store_checkpoint(sess, '%s/checkpoint_best/checkpoint' % args.model_dir, "best")

						previous_losses = previous_losses[1:] + [loss_list[0]]
						loss_list *= .0
						time_step = .0

				batched_data = data.get_next_batch("train")

			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")

	def test_process(self, sess, data, args):
		def get_batch_results(batched_responses_id, data):
			batch_results = []
			for response_id in batched_responses_id:
				response_id_list = response_id.tolist()
				if data.eos_id in response_id_list:
					end = response_id_list.index(data.eos_id) + 1
					result_id = response_id_list[:end]
				else:
					result_id = response_id_list
				batch_results.append(result_id)
			return batch_results

		def padding(matrix, pad_go_id=False):
			l = max([len(d) for d in matrix])
			if not pad_go_id:
				res = [[d + [data.pad_id] * (l - len(d)) for d in matrix]]
			else:
				res = [[[data.go_id] + d + [data.pad_id] * (l - len(d)) for d in matrix]]
			return res

		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")
		cnt = 0
		start_time = time.time()
		while batched_data != None:
			conv_data = [{'contexts': [], 'responses': [], 'generations': []} for _ in range(len(batched_data['turn_length']))]
			for cut_batch_data in self.split_session(batched_data, args.session_window, inference=True):
				eval_out = self.step_decoder(sess, cut_batch_data, forward_only=True)
				decoder_loss, gen_prob = eval_out[:6], eval_out[-1]
				batched_responses_id = self.step_decoder(sess, cut_batch_data, inference=True)[0]
				batch_results = get_batch_results(batched_responses_id, data)

				cut_batch_data['gen_prob'] = gen_prob
				cut_batch_data['generations'] = batch_results
				responses_length = []
				for length in cut_batch_data['responses_length']:
					if length == 1:
						length += 1
					responses_length.append(length)
				metric1_data = {
						'sent_allvocabs': np.expand_dims(cut_batch_data['responses'], 1),
						'sent_length': np.expand_dims(responses_length, 1),
						'multi_turn_gen_log_prob': np.expand_dims(cut_batch_data['gen_prob'], 1)
						}
				metric1.forward(metric1_data)
				valid_index = [idx for idx, length in 
						enumerate(cut_batch_data['responses_length']) if length > 1]
				for key in ['contexts', 'responses', 'generations']:
					for idx, d in enumerate(cut_batch_data[key]):
						if idx in valid_index:
							if key == 'contexts':
								d = d[cut_batch_data['contexts_length'][idx] - 1]
							conv_data[idx][key].append(list(d))

				if (cnt + 1) % 10 == 0:
					print('processing %d batch data, time cost %.2f s/batch' % (cnt, (time.time() - start_time) / 10))
					start_time = time.time()
				cnt += 1

			for conv in conv_data:
				metric2_data = {
						#'context_allvocabs': np.array(padding(conv['contexts'])),
						'turn_length': np.array([len(conv['contexts'])], dtype=np.int32),
						'sent_allvocabs':np.array(padding(conv['responses'])),
						'multi_turn_gen': np.array(padding(conv['generations'], pad_go_id=True))
						}
				metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		res = metric1.close()
		res.update(metric2.close())

		test_file = args.out_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'a') as f:
			print("Test Result:")
			for key, value in res.items():
				if isinstance(value, float) or isinstance(value, int):
					print("\t%s:\t%f" % (key, value))
					f.write("%s:\t%f\n" % (key, value))
			for i, context in enumerate(res['context']):
				f.write("session: \t%d\n" % i)
				for j in range(len(context)):
					f.write("\tpost:\t%s\n" % " ".join(context[j]))
					f.write("\tresp:\t%s\n" % " ".join(res['reference'][i][j]))
					f.write("\tgen:\t%s\n\n" % " ".join(res['gen'][i][j]))
				f.write("\n")

		print("result output to %s" % test_file)
		return {key: val for key, val in res.items() if key not in ['context', 'reference', 'gen']}

	def test_multi_ref(self, sess, data, word2vec, args):
		def process_cands(candidates):
			res = []
			for cands in candidates:
				tmp = []
				for sent in cands:
					tmp.append([data.go_id] + \
						[wid if wid < data.vocab_size else data.unk_id for wid in sent] + [data.eos_id])
				res.append(tmp)
			return res
		prec_rec_metrics = data.get_multi_ref_metric(generated_num_per_context=args.repeat_N, word2vec=word2vec)
		for batch_data in self.multi_reference_batches(data, args.batch_size):
			responses = []
			for _ in range(args.repeat_N):
				batched_responses_id = self.step_decoder(sess, batch_data, inference=True)[0]
				for rid, resp in enumerate(batched_responses_id):
					resp = list(resp)
					if rid == len(responses):
						responses.append([])
					# if data.eos_id in resp:
					# 	resp = resp[:resp.index(data.eos_id)]
					resp = data.trim_index(resp)
					if len(resp) == 0:
						resp = [data.unk_id]
					responses[rid].append(resp + [data.eos_id])
			metric_data = {'candidate_allvocabs': process_cands(batch_data['candidate_allvocabs']), 'multiple_gen_key': responses}
			prec_rec_metrics.forward(metric_data)

		res = prec_rec_metrics.close()

		test_file = args.out_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'w') as f:
			print("Test Multi Reference Result:")
			f.write("Test Multi Reference Result:\n")
			for key, val in res.items():
				if isinstance(val, float) or isinstance(val, int):
					print("\t{}\t{}".format(key, val))
					f.write("\t{}\t{}".format(key, val) + "\n")
			f.write("\n")
		return res
