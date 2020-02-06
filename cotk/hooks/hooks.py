'''
A utils providing callback hooks.
'''

from inspect import signature
from functools import wraps
import json
import copy
import weakref
import pkg_resources

#pylint: disable=global-statement

hooks_listener = []

def invoke_listener(method, *argv):
	r'''invoke listener with method'''
	global hooks_listener
	for listener in hooks_listener:
		getattr(listener, method)(*argv)

def compress_dict(dic):
	r'''copy a dict and prepare converting to json. If a item is too big, use ... instead.'''
	res = {}

	def peek_json_length(obj):
		try:
			return len(json.dumps(obj))
		except TypeError:
			return -1

	for key, value in dic.items():
		if peek_json_length(value) in range(0, 50):
			res[key] = copy.deepcopy(value)
		else:
			if isinstance(value, list):
				res[key] = "[...]"
			elif isinstance(value, dict):
				res[key] = "{...}"
			else:
				res[key] = "..."
	return res

def hook_dataloader(fn):
	r'''decorator for dataloader.__init___'''
	sign = signature(fn)
	@wraps(fn)
	def wrapped(*args, **kwargs):
		bound = sign.bind(*args, **kwargs)
		bound.apply_defaults()
		bound = dict(bound.arguments)
		self = bound['self']
		del bound['self']
		invoke_listener("add_dataloader", self, fn.__qualname__.split(".")[0], bound)
		return fn(*args, **kwargs)

	return wrapped

def hook_metric(fn):
	r'''decorator for metric.__init__'''
	sign = signature(fn)
	@wraps(fn)
	def wrapped(*args, **kwargs):
		bound = sign.bind(*args, **kwargs)
		bound.apply_defaults()
		bound = dict(bound.arguments)
		self = bound['self']
		del bound['self']
		invoke_listener("add_metric", self, fn.__qualname__.split(".")[0], bound)
		return fn(*args, **kwargs)
	return wrapped

def hook_metric_close(fn):
	r'''decorator for metric.close'''
	sign = signature(fn)
	@wraps(fn)
	def wrapped(*args, **kwargs):
		bound = sign.bind(*args, **kwargs)
		bound.apply_defaults()
		bound = dict(bound.arguments)
		self = bound['self']
		return_dict = fn(*args, **kwargs)
		invoke_listener("invoke_metric_close", self, return_dict)
		return return_dict
	return wrapped

def hook_wordvec(fn):
	r'''decorator for wordvec.__init__'''
	sign = signature(fn)
	@wraps(fn)
	def wrapped(*args, **kwargs):
		bound = sign.bind(*args, **kwargs)
		bound.apply_defaults()
		bound = dict(bound.arguments)
		self = bound['self']
		del bound['self']
		invoke_listener("add_wordvec", self, fn.__qualname__.split(".")[0], bound)
		return fn(*args, **kwargs)
	return wrapped

class BaseHooksListener:
	r'''An abstract class implement the basic hook listener'''
	def add_dataloader(self, obj, dataloader, args):
		r'''invoked at dataloader.__init__

		Arguments:
			obj (Dataloader): the obj created.
			dataloader (str): the string of obj's class
			args (tuple): the arguments of dataloader.__init__ except self
		'''
		pass

	def add_metric(self, obj, metric, args):
		r'''invoked at metric.__init__

		Arguments:
			obj (:class:`.metric.MetricBase`): the obj created.
			metric (str): the string of obj's class
			args (tuple): the arguments of metric.__init__ except self
		'''
		pass

	def invoke_metric_close(self, obj, return_dict):
		r'''invoked at metric.close

		Arguments:
			obj (:class:`.metric.MetricBase`): the obj of metric
			args (tuple): the arguments of metric.close except self
		'''
		pass

	def add_wordvec(self, obj, wordvec, args):
		r'''invoked at wordvec.__init__

		Arguments:
			obj (:class:`.wordvec.WordVector`): the obj created.
			wordvec (str): the string of obj's class
			args (tuple): the arguments of wordvec.__init__ except self
		'''
		pass

class SimpleHooksListener(BaseHooksListener):
	r'''An simple recorder'''
	def __init__(self):
		self.record = {
			"cotk_version": pkg_resources.require("cotk")[0].version,
			"dataloader": [],
			"wordvec": []
		}
		self.dataloader_set = weakref.WeakKeyDictionary()
		self.metric_set = weakref.WeakKeyDictionary()
		self.hash_set = {}

	def close(self, result_dic):
		r'''invoked at dataloader.__init__

		Returns:

			dict. At least containing:

			* cotk_version (str)
			* dataloader (list): each element is (dataset_args, [metric_args])
			* wordvec (list): each element is wordvec_args
		'''
		for key, value in result_dic.items():
			if "hashvalue" not in key:
				continue
			try:
				dataset_args, metric_args = self.hash_set[key + value]
			except KeyError:
				print("WARNING: Unknown hashvalue for hooks. " +
					"Cotk can't fetch the metric information about %s:%s. " % (key, value) +
					"It can be caused by an unhooked metric.close.")
				continue
			for dataloader, metric in self.record['dataloader']:
				if id(dataloader) == id(dataset_args):
					metric.append(metric_args)
					break
			else:
				self.record['dataloader'].append((dataset_args, [metric_args]))
		return self.record

	def add_dataloader(self, obj, dataloader, args):
		args = compress_dict(args)
		args['clsname'] = dataloader
		self.dataloader_set[obj] = args

	def add_metric(self, obj, metric, args):
		dataloader = args['dataloader']
		del args['dataloader']
		args = compress_dict(args)
		args['clsname'] = metric
		args['dataloader'] = dataloader
		self.metric_set[obj] = args

	def invoke_metric_close(self, obj, return_dict):
		for key, value in return_dict.items():
			if "hashvalue" not in key:
				continue
			try:
				metric_args = self.metric_set[obj]
			except KeyError:
				print("WARNING: Unknown metrics for hooks. " +
					"Cotk can't fetch the metric information about %s. " % (obj) +
					"It can be caused by an unhooked metric.")
				continue
			try:
				dataset_args = self.dataloader_set[metric_args['dataloader']]
			except KeyError:
				print("WARNING: Unknown dataloader for hooks. " +
					"Cotk can't fetch the dataloader information about %s. " % (metric_args['dataloader']) +
					"It can be caused by an unhooked dataloader.")
				continue
			metric_args = {key: value for key, value in self.metric_set[obj].items() if key != "dataloader"}
			self.hash_set[key + value] = (dataset_args, metric_args)

	def add_wordvec(self, obj, wordvec, args):
		args = compress_dict(args)
		args['clsname'] = wordvec
		self.record['wordvec'].append(args)

def start_recorder():
	r'''Start recorder'''
	global hooks_listener
	hooks_listener.clear()
	hooks_listener.append(SimpleHooksListener())

def close_recorder(result_dict):
	r'''Close recorder and return the recorded information.'''
	global hooks_listener
	assert len(hooks_listener) == 1
	return hooks_listener[0].close(result_dict)
