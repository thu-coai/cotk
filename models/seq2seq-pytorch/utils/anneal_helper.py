#coding:utf-8

#TODO: make it to a hook, not hardcode in model_helper
class AnnealHelper:
	def __init__(self, instance, name, beginValue, startBatch, startValue, endValue, multi):
		self.instance = instance
		self.name = name
		self.initial = startValue
		self.end = endValue
		self.startbatch = startBatch
		self.multi = multi
		self.beginValue = beginValue

	def step(self):
		if self.instance.now_batch >= self.startbatch and self.instance.param.other_weights[self.name] == self.beginValue:
			self.instance.param.other_weights[self.name] = self.initial

		if self.instance.now_batch > self.startbatch:
			self.instance.param.other_weights[self.name] *= self.multi
			if self.multi > 1:
				if self.instance.param.other_weights[self.name] > self.end:
					self.instance.param.other_weights[self.name] = self.end
			else:
				if self.instance.param.other_weights[self.name] < self.end:
					self.instance.param.other_weights[self.name] = self.end

	def over(self):
		return self.instance.param.other_weights[self.name] == self.end

class AnnealParameter(tuple):
	'''Set parameter for AnnealHelper. BaseModel will automatically create AnnealHelper.
	'''
	@staticmethod
	def create_set(value):
		return AnnealParameter(("set", {"value": value}))

	@staticmethod
	def create_hold(value):
		return AnnealParameter(("hold", {"value": value}))

	@staticmethod
	def create_anneal(beginValue, startBatch, startValue, endValue, multi):
		return AnnealParameter(("anneal", {"beginValue": beginValue, "startBatch": startBatch, "startValue": startValue, "endValue":endValue, "multi":multi}))

	@staticmethod
	def create_set_and_anneal(startValue, endValue, multi):
		return AnnealParameter(("set&anneal", {"startValue": startValue, "endValue":endValue, "multi":multi}))
