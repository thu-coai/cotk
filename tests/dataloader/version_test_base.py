import os
import json
import pathlib
import inspect
import copy
import functools

from cotk.dataloader import Dataloader

cur_dir = pathlib.Path(__file__).parent.absolute()
version_dir = cur_dir / 'version_test_data'


class PatchDataloader:
	"""
	Decorator a dataloader class and record arguments and hash value when the dataloader's `__init__` is called.

	Examples:
		>>> from cotk.dataloader import OpenSubtitles
		>>> PatchedSwitchboardCorpus = PatchDataloader(OpenSubtitles)
		>>> PatchedSwitchboardCorpus("./tests/dataloader/dummy_opensubtitles#OpenSubtitles")
		>>> PatchedSwitchboardCorpus("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", invalid_vocab_times=1)
		>>> PatchedSwitchboardCorpus.save()
	"""

	def __init__(self, dl_class):
		functools.update_wrapper(self, dl_class)
		self.__sign = inspect.signature(dl_class)
		self.__version_info = []

	def __call__(self, *args, **kwargs):
		bound = self.__sign.bind(*args, **kwargs)
		bound.apply_defaults()
		dl = self.__wrapped__(*args, **kwargs)
		self.__version_info.append({
			'args': copy.deepcopy(bound.args),
			'kwargs': copy.deepcopy(bound.kwargs),
			'hash_value': dl.hash_value
		})
		return dl

	def save(self, version_dir=str(version_dir)):
		version_path = os.path.join(version_dir,
									'{}_v{}.jsonl'.format(self.__wrapped__.__name__, self.__wrapped__._version))
		with open(version_path, 'w', encoding='utf-8') as out:
			for item in self.__version_info:
				out.write(json.dumps(item))
				out.write('\n')


def load_version_info(version_path):
	assert os.path.isfile(version_path), '`{}` is not a file.'.format(version_path)
	with open(version_path, 'r', encoding='utf-8') as fin:
		return [json.loads(line) for line in fin if line.strip()]


def base_test_version(dl_class):
	"""

	Args:
		dl_class (type): subclass of Dataloader.
	"""
	if isinstance(dl_class, str):
		dl_class = Dataloader.load_class(dl_class)
	assert hasattr(dl_class, '_version')
	version = dl_class._version
	version_path = str(version_dir / '{}_v{}.jsonl'.format(dl_class.__name__, version))
	version_info = load_version_info(version_path)
	assert version_info
	for dic in version_info:
		assert 'setting_hash' in dic
		assert 'vocab_hash' in dic
		assert 'raw_data_hash' in dic
		assert 'data_hash' in dic
		assert 'general_hash' in dic
		
		assert 'args' in dic
		assert 'kwargs' in dic
		
		args = dic['args']
		kwargs = dic['kwargs']
		dl = dl_class(*args, **kwargs)
		assert dic['setting_hash'] == dl.get_setting_hash()
		assert dic['vocab_hash'] == dl.get_vocab_hash()
		assert dic['raw_data_hash'] == dl.get_raw_data_hash()
		assert dic['data_hash'] == dl.get_data_hash()
		assert dic['general_hash'] == dl.get_general_hash()
