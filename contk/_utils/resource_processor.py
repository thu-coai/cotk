'''Processor for resource after download and before read
'''
class ResourceProcessor:
	'''Base class for processor.
	'''
	def __init__(self):
		pass

class DefaultResourceProcessor(ResourceProcessor):
	'''Processor for default resource: do nothing.
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return local_path

	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		return local_path
