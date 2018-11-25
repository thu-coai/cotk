import pickle
import os

def try_cache(module, args, name=None):
	if name is None:
		name = module.__name__
	if not os.path.exists("cache"):
		os.makedirs("cache")
	fname = "cache/%s.pkl" % name
	if os.path.exists(fname):
		f = open(fname, "rb")
		info, obj = pickle.load(f)
		f.close()
	else:
		info = None
		obj = None
	if info is not None:
		assert info == repr(args)
	if info != repr(args):
		obj = module(*args)
		f = open(fname, "wb")
		pickle.dump((repr(args), obj), f, -1)
		f.close()
	return obj
