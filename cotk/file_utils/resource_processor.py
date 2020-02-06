'''Processor for resource after download and before read
'''
import os
import zipfile
import shutil
import json
from itertools import chain
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

class ZipResourceProcessor(ResourceProcessor):
	'''Processor for default resource: extract zip.
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		if os.path.isdir(local_path):
			return local_path
		dst_dir = local_path + '_unzip'
		unzip_file(local_path, dst_dir)
		return dst_dir

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

	def postprocess(self, local_path):
		local_path = super().postprocess(local_path)
		new_local_path = os.path.join(local_path, 'processed')
		os.makedirs(new_local_path, exist_ok=True)
		for key in ['train', 'dev', 'test']:
			local_file = os.path.join(local_path, 'mscoco_%s.txt' % key)
			new_local_file = os.path.join(new_local_path, '%s.txt' % key)
			if os.path.isfile(local_file):
				shutil.copy(local_file, new_local_file)
		return new_local_path

class OpenSubtitlesResourceProcessor(BaseResourceProcessor):
	'''Processor for OpenSubtitles Dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'opensubtitles')

	def postprocess(self, local_path):
		local_path = super().postprocess(local_path)
		new_local_path = os.path.join(local_path, 'processed')
		os.makedirs(new_local_path, exist_ok=True)
		for key in ['train', 'test', 'dev']:
			post_path = os.path.join(local_path, 'opensub_pair_%s.post' % key)
			response_path = os.path.join(local_path, 'opensub_pair_%s.response' % key)
			if not os.path.isfile(post_path) or not os.path.isfile(response_path):
				continue
			with open(post_path, 'r', encoding='utf-8') as posts:
				with open(response_path, 'r', encoding='utf-8') as responses:
					with open(os.path.join(new_local_path, '%s.txt' % key), 'w', encoding='utf-8') as out:
						for post, resp in zip(posts, responses):
							out.write(post if post[-1] == '\n' else (post + '\n'))
							out.write(resp if resp[-1] == '\n' else (resp + '\n'))
		return new_local_path

class UbuntuResourceProcessor(BaseResourceProcessor):
	'''Processor for UbuntuCorpus dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'ubuntu_dataset')

	def postprocess(self, local_path):
		import csv
		local_path = super().postprocess(local_path)
		new_local_path = os.path.join(local_path, 'processed')
		os.makedirs(new_local_path, exist_ok=True)
		for key in ['train', 'dev', 'test']:
			local_file = os.path.join(local_path, 'ubuntu_corpus_%s.csv' % key)
			if not os.path.isfile(local_file):
				continue
			new_local_file = os.path.join(new_local_path, '%s.txt' % key)
			with open(local_file, 'r', encoding='utf-8') as f:
				reader = csv.reader(f)
				head = next(reader)
				if head[2] == 'Label':
					raw_data = [d[0] + d[1] for d in reader if d[2] == '1.0']
				else:
					raw_data = [d[0] + d[1] for d in reader]

			with open(new_local_file, 'w', encoding='utf-8') as f:
				for session in raw_data:
					for sent in session.strip().replace('__eou__', '').split('__eot__'):
						f.write(sent)
						f.write('\n')
					f.write('\n')
		return new_local_path


class SwitchboardCorpusResourceProcessor(BaseResourceProcessor):
	'''Processor for SwitchboardCorpus dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'switchboard_corpus')

	def postprocess(self, local_path):
		local_path = super().postprocess(local_path)
		new_local_path = os.path.join(local_path, 'processed')
		os.makedirs(new_local_path, exist_ok=True)

		for key in ['train', 'test', 'dev', 'multi_ref']:
			filepath = os.path.join(local_path, 'switchboard_corpus_%s.jsonl' % key)
			new_filepath = os.path.join(new_local_path, '%s.txt' % key)
			res = self._read_file(filepath, key == 'multi_ref')

			with open(new_filepath, 'w', encoding='utf-8') as fout:
				if key != 'multi_ref':
					dataset = res  # sessions
				else:
					sessions, responses = res
					dataset = chain.from_iterable(zip(sessions, responses))  # [session1, response1, session2, response2, ...]
					# response is like a session. Both contain several sentences.
				for sess in dataset:
					assert sess
					for line in sess:
						if line[-1] != '\n':
							line += '\n'
						fout.write(line)
					fout.write('\n')
		return new_local_path

	def _read_file(self, filepath, read_multi_ref=False):
		"""
		Arguments:
			filepath (str): Name of the file to read from
			read_multi_ref (bool):
				If False, add turn ``<d>`` ahead of each session
				If True, add turn ``<d>`` at the end of each session and read candidate ``responses``
		"""
		sessions = []
		if read_multi_ref:
			responses = []
		with open(filepath, "r", encoding='utf-8') as data_file:
			for line in data_file:
				line = json.loads(line)
				prefix_utts = [['X', '<d>']] + line['utts']
				# pylint: disable=cell-var-from-loop
				suffix_utts = list(map(lambda utt: utt[1][1].strip() + ' ' \
							if prefix_utts[utt[0]][0] == utt[1][0] \
							else '<eos> ' + utt[1][1].strip() + ' ', enumerate(line['utts'])))
				utts = ('<d> ' + "".join(suffix_utts).strip()).split("<eos>")
				sess = utts[1:] + ['<d>'] if read_multi_ref else utts
				sessions.append(sess)

				if read_multi_ref:
					responses.append([resp for _, resp in line['responses']])
		if read_multi_ref:
			return sessions, responses
		else:
			return sessions


class SSTResourceProcessor(BaseResourceProcessor):
	'''Processor for SST dataset
	'''
	def preprocess(self, local_path):
		'''Preprocess after download and before save.
		'''
		return self.basepreprocess(local_path, 'trees')

	def _parseline(self, line):
		label = int(line[1])
		line = line.split(')')
		sent = [x.split(' ')[-1].lower() for x in line if x != '']
		return label, ' '.join(sent)

	def _postprocess(self, src, dest, key):
		with open(os.path.join(src, key + '.txt'), 'r', encoding='utf-8') as fp:
			labels, sents = [], []
			for label, sent in map(self._parseline, fp):
				labels.append(label)
				sents.append(sent)
		with open(os.path.join(dest, key + '.txt'), 'w', encoding='utf-8') as fp:
			fp.writelines(sents)
		with open(os.path.join(dest, key + '_labels.json'), 'w', encoding='utf-8') as fp:
			json.dump(labels, fp, ensure_ascii=False)

	def postprocess(self, local_path):
		local_path = super().postprocess(local_path)
		new_local_path = os.path.join(local_path, 'processed')
		os.makedirs(new_local_path, exist_ok=True)

		for key in ['train', 'test', 'dev']:
			if not os.path.isfile(os.path.join(local_path, key + '.txt')):
				raise FileNotFoundError("there isn\'t %s in %s" % (key + '.txt', local_path))
			else:
				self._postprocess(local_path, new_local_path, key)
		return new_local_path

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
