import os

import json
import numpy as np
import tensorflow as tf
from cotk.dataloader import MultiTurnDialog
from cotk.wordvector import WordVector, Glove
from utils import debug, try_cache

from model import HredModel

def create_model(sess, data, args, embed):
	with tf.variable_scope(args.name):
		model = HredModel(data, args, embed)
		model.print_parameters()
		latest_dir = '%s/checkpoint_latest' % args.model_dir
		best_dir = '%s/checkpoint_best' % args.model_dir
		if tf.train.get_checkpoint_state(latest_dir) and args.restore == "last":
			print("Reading model parameters from %s" % latest_dir)
			model.latest_saver.restore(sess, tf.train.latest_checkpoint(latest_dir))
		else:
			if tf.train.get_checkpoint_state(best_dir) and args.restore == "best":
				print('Reading model parameters from %s' % best_dir)
				model.best_saver.restore(sess, tf.train.latest_checkpoint(best_dir))
			else:
				print("Created model with fresh parameters.")
				global_variable = [gv for gv in tf.global_variables() if args.name in gv.name]
				sess.run(tf.variables_initializer(global_variable))

	return model


def main(args):
	if args.debug:
		debug()

	if args.cuda:
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
	else:
		config = tf.ConfigProto(device_count={'GPU': 0})
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	data_class = MultiTurnDialog.load_class(args.dataset)
	wordvec_class = WordVector.load_class(args.wvclass)
	if wordvec_class == None:
		wordvec_class = Glove
	if args.cache:
		data = try_cache(data_class, (args.datapath,), args.cache_dir)
		vocab = data.vocab_list
		embed = try_cache(lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl),
						  (args.wvpath, args.embedding_size, vocab),
						  args.cache_dir, wordvec_class.__name__)
	else:
		data = data_class(args.datapath, 
				min_vocab_times=args.min_vocab_times,
				max_sent_length=args.max_sent_length, 
				max_turn_length=args.max_turn_length)
		wv = wordvec_class(args.wvpath)
		vocab = data.vocab_list
		embed = wv.load_matrix(args.embedding_size, vocab)

	embed = np.array(embed, dtype = np.float32)

	with tf.Session(config=config) as sess:
		model = create_model(sess, data, args, embed)
		if args.mode == "train":
			model.train_process(sess, data, args)
		else:
			test_res = model.test_process(sess, data, args)

			for key, val in test_res.items():
				if isinstance(val, bytes):
					test_res[key] = str(val)
			json.dump(test_res, open("./result.json", "w"))