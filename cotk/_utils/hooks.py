'''
A utils providing callback hooks.
'''

from inspect import signature
import pkg_resources

#pylint: disable=global-statement

hooks_listener = []

def invoke_listener(method, *argv):
	r'''invoke listener with method'''
	global hooks_listener
	for listener in hooks_listener:
		getattr(listener, method)(*argv)

def hook_dataloader(fn):
	r'''decorator for dataloader.__init___'''
	sign = signature(fn)
	def wrapped(*args, **kwargs):
		binded = sign.bind(*args, **kwargs)
		binded.apply_defaults()
		binded = dict(binded.arguments)
		self = binded['self']
		del binded['self']
		invoke_listener("add_dataloader", self, fn.__qualname__.split(".")[0], binded)
		return fn(*args, **kwargs)
	return wrapped

def hook_standard_metric(metric_name=""):
	r'''decorator for dataloader.get_metric'''
	def decorator(fn):
		sign = signature(fn)
		def wrapped(*args, **kwargs):
			binded = sign.bind(*args, **kwargs)
			binded.apply_defaults()
			binded = dict(binded.arguments)
			self = binded['self']
			del binded['self']
			invoke_listener("add_standard_metric", self, fn.__qualname__.split(".")[0], metric_name, binded)
			print(kwargs)
			return fn(*args, **kwargs)
		return wrapped
	return decorator

def hook_wordvec(fn):
	r'''decorator for wordvec.__init__'''
	sign = signature(fn)
	def wrapped(*args, **kwargs):
		binded = sign.bind(*args, **kwargs)
		binded.apply_defaults()
		binded = dict(binded.arguments)
		self = binded['self']
		del binded['self']
		invoke_listener("add_wordvec", self, fn.__qualname__.split(".")[0], binded)
		return fn(*args, **kwargs)
	return wrapped

class BaseHooksListener:
	r'''An abstract class implement the basic hook listener'''
	def add_dataloader(self, dataloader, args):
		pass

	def add_standard_metric(self, dataloader, metric_type, args):
		pass

	def add_wordvec(self, wordvec, args):
		pass

class SimpleHooksListener(BaseHooksListener):
	r'''An simple recorder'''
	def __init__(self):
		self.record = {
			"cotk_version": pkg_resources.require("cotk")[0].version,
			"dataloader": {},
			"standard_metric": {},
			"wordvec": {}
		}
		self.dataloader_id = {}

	def close(self):
		return self.record

	def add_dataloader(self, obj, dataloader, args):
		id = len(self.dataloader_id)
		args["id"] = id
		self.record["dataloader"][dataloader] = args
		print(obj)
		self.dataloader_id[obj] = id
		print(self.dataloader_id)

	def add_standard_metric(self, obj, clsname, metric_type, args):
		print(obj)
		print(self.dataloader_id)
		args['dataloader_id'] = self.dataloader_id[obj]
		self.record["standard_metric"][clsname + "_" + metric_type] = args

	def add_wordvec(self, obj, wordvec, args):
		self.record["wordvec"][wordvec] = args

def start_recorder():
	r'''Start recorder'''
	global hooks_listener
	hooks_listener.clear()
	hooks_listener.append(SimpleHooksListener())

def close_recorder():
	r'''Close recorder and return the recorded information.'''
	global hooks_listener
	assert len(hooks_listener) == 1
	return hooks_listener[0].close()
