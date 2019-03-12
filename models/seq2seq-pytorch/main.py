# coding:utf-8
import logging
import json

from cotk.dataloader import SingleTurnDialog
from cotk.wordvector import WordVector, Glove

from utils import debug, try_cache, cuda_init, Storage
from seq2seq import Seq2seq

def main(args):
	logging.basicConfig(\
		filename=0,\
		level=logging.DEBUG,\
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',\
		datefmt='%H:%M:%S')

	if args.debug:
		debug()
	logging.info(json.dumps(args, indent=2))

	cuda_init(0, args.cuda)

	volatile = Storage()
	data_class = SingleTurnDialog.load_class(args.dataset)
	wordvec_class = WordVector.load_class(args.wvclass)
	if wordvec_class is None:
		wordvec_class = Glove
	if args.cache:
		dm = try_cache(data_class, (args.datapath,), args.cache_dir)
		volatile.wordvec = try_cache(\
			lambda wv, ez, vl: wordvec_class(wv).load(ez, vl), \
			(args.wvpath, args.embedding_size, dm.vocab_list), 
			args.cache_dir, wordvec_class.__name__)
	else:
		dm = data_class(args.datapath)
		wv = wordvec_class(args.wvpath)
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
		raise ValueError("Unknown mode")
