#coding:utf-8

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

def createSet(value):
	return ("set", {"value": value})

def createHold(value):
	return ("hold", {"value": value})

def createAnneal(beginValue, startBatch, startValue, endValue, multi):
	return ("anneal", {"beginValue": beginValue, "startBatch": startBatch, "startValue": startValue, "endValue":endValue, "multi":multi})

def createSetAndAnneal(startValue, endValue, multi):
	return ("set&anneal", {"startValue": startValue, "endValue":endValue, "multi":multi})

