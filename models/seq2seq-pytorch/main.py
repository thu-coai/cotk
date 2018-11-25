# coding:utf-8
import logging

from cotk.dataloader import OpenSubtitles
from cotk.wordvector import Glove

from utils import debug, try_cache, cuda_init, Storage
from seq2seq import Seq2seq

def main(args):
	logging.basicConfig(\
		filename = 0,\
		level=logging.DEBUG,\
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',\
		datefmt='%H:%M:%S')

	if args.debug:
		debug()
	logging.info(args)

	cuda_init(0, args.cuda)

	volatile = Storage()
	if args.cache:
		dm = try_cache(OpenSubtitles, (args.dataset,))
		volatile.wordvec = try_cache(\
			lambda wv, ez, vl: Glove(wv).load(ez, vl), \
			(args.wordvec, args.embedding_size, dm.vocab_list), "wordvec")
	else:
		dm = OpenSubtitles(args.dataset)
		wv = Glove(args.wordvec)
		volatile.wordvec = wv.load(args.embedding_size, dm.vocab_list)

	volatile.dm = dm

	param = Storage()
	param.args = args
	param.volatile = volatile

	model = Seq2seq(param)
	if args.mode == "train":
		model.train_process()
	elif args.mode == "test":
		model.test_process()
	else:
		raise ValueError("Unkown mode")
