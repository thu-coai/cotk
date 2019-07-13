"""
Utilities for working with the local dataset cache.
"""

from ._utils.file_utils import load_file_from_url as _load_file_from_url

r"""
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
""" #pylint: disable=pointless-string-statement

def load_file_from_url(url, force=False, cache_dir=None):
	'''Download a file from the given ``url``. If the file has been downloaded, it will be
	cached in ``cache_dir``. However, the function can't check whether the file from ``url``
	is changed online. If you have the wrong cache,	you may manually delete the file cached
	located by the return value.

	Arguments:
		url(str): A url indicating the file online.
		force(bool): Force to download and ignore the existing file. Default: ``False``
		cache_dir(str, optional): A path indicating where the cache place.
			Default: if ``None``, a default cache path is used.

	Returns:
		(str) The local path of downloaded model.

	Example:
		>>> load_model_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
		>>> {CACHE_DIR}/resnet18-5c106cde.pth
	'''

	if cache_dir is not None:
		return _load_file_from_url(url, force, cache_dir=cache_dir)
	else:
		return _load_file_from_url(url, force)
