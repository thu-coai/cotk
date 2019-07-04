'''Processor for resource after download and before read
'''
import os
import zipfile
import shutil
from .metaclass import LoadClassInterface

def unzip_file(src_path, dst_dir):
	'''unzip the zip file in src_path to dst_dir
	'''
	if zipfile.is_zipfile(src_path):
		with zipfile.ZipFile(src_path, 'r') as zip_obj:
			zip_obj.extractall(dst_dir)
	else:
		raise ValueError('{} is not zip'.format(src_path))

class ResourceProcessor(LoadClassInterface):
	'''Base class for processor.
	'''
	def __init__(self, cache_dir=None, config_dir=None):
		self.cache_dir = cache_dir
		self.config_dir = config_dir

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

class BaseResourceProcessor(ResourceProcessor):
	"""Basic processor for MSCOCO, OpenSubtitles, Ubuntu..."""
	def basepreprocess(self, local_path, name):
		'''Preprocess after download and before save.
		'''
		if os.path.isdir(local_path):
			return local_path
		dst_dir = local_path + '_unzip'
		unzip_file(local_path, dst_dir)
		return os.path.join(dst_dir, name)

	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		return local_path

#TODO: merge the following Processor because of duplicate codes
class MSCOCOResourceProcessor(BaseResourceProcessor):
	'''Processor for MSCOCO dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'mscoco')

class OpenSubtitlesResourceProcessor(BaseResourceProcessor):
	'''Processor for OpenSubtitles Dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'opensubtitles')

class UbuntuResourceProcessor(BaseResourceProcessor):
	'''Processor for UbuntuCorpus dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'ubuntu_dataset')

class SwitchboardCorpusResourceProcessor(BaseResourceProcessor):
	'''Processor for SwitchboardCorpus dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'switchboard_corpus')

class GloveResourceProcessor(ResourceProcessor):
	'''Base Class for all dimension version of glove wordvector.
	'''
	def __init__(self, cache_dir=None, config_dir=None):
		super(GloveResourceProcessor, self).__init__(cache_dir, config_dir)
		self.other_gloves = []

	def basepreprocess(self, local_path, name):
		'''Preprocess after download and before save.
		'''
		dst_dir = local_path + '_unzip'
		unzip_file(local_path, dst_dir)
		filenames = os.listdir(dst_dir)
		for filename in filenames:
			if os.path.isdir(os.path.join(dst_dir, filename)):
				continue
			dim = filename.split('.')[-2]
			if dim != name and self.cache_dir is not None and self.config_dir is not None:
				self.other_gloves.append(["resources://Glove%s" % (dim), \
										local_path, self.cache_dir, self.config_dir])
				continue
			sub_dir = os.path.join(dst_dir, dim)
			os.makedirs(sub_dir, exist_ok=True)
			os.rename(os.path.join(dst_dir, filename), os.path.join(sub_dir, 'glove.txt'))
		return dst_dir

	def basepostprocess(self, local_path, name):
		'''Postprocess before read.
		'''
		from .file_utils import import_local_resources
		for glove in self.other_gloves:
			import_local_resources(*glove, ignore_exist_error=True)
		self.other_gloves = []
		return os.path.join(local_path, name)

class Glove50dResourceProcessor(GloveResourceProcessor):
	'''Processor for glove50d wordvector
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, '50d')

	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		return self.basepostprocess(local_path, '50d')

class Glove100dResourceProcessor(GloveResourceProcessor):
	'''Processor for glove100d wordvector
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, '100d')

	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		return self.basepostprocess(local_path, '100d')

class Glove200dResourceProcessor(GloveResourceProcessor):
	'''Processor for glove200d wordvector
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, '200d')

	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		return self.basepostprocess(local_path, '200d')

class Glove300dResourceProcessor(GloveResourceProcessor):
	'''Processor for glove300d wordvector
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, '300d')

	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		return self.basepostprocess(local_path, '300d')
