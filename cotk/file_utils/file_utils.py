"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import json
import tempfile
import shutil
import hashlib
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse
import requests

from tqdm import tqdm
from checksumdir import dirhash

from .resource_processor import ResourceProcessor

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)
CACHE_DIR = os.path.join(str(Path.home()), '.cotk_cache')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../resource_config')

def _url_to_filename(url):
	r'''Convert the url to sha256 as filename
	'''
	url_bytes = url.encode('utf-8')
	url_hash = hashlib.sha256(url_bytes)
	filename = url_hash.hexdigest()

	return filename


def _get_config(res_name, config_dir=None):
	'''Get config(dict) by the name of resource'''
	config_dir = config_dir or CONFIG_DIR
	config_path = os.path.join(config_dir, res_name + '.json')
	if not os.path.exists(config_path):
		raise FileNotFoundError("file {} not found".format(config_path))
	with open(config_path, 'r', encoding='utf-8') as config_file:
		config = json.load(config_file)

	return config


def _http_get(url, temp_file):
	'''Pull a file directly from http'''
	req = requests.get(url, stream=True)
	content_length = req.headers.get('Content-Length')
	total = int(content_length) if content_length is not None else None
	progress = tqdm(unit="B", total=total)
	for chunk in req.iter_content(chunk_size=65536):
		if chunk:
			progress.update(len(chunk))
			temp_file.write(chunk)
	progress.close()


def _get_file_sha256(file_path):
	'''Get sha256 of given file'''
	hash_sha256 = hashlib.sha256()
	with open(file_path, "rb") as fin:
		for chunk in iter(lambda: fin.read(4096), b""):
			hash_sha256.update(chunk)
	return hash_sha256.hexdigest()


def _get_hashtag(file_path):
	'''Get sha256 of given directory or file'''
	if os.path.isdir(file_path):
		return dirhash(file_path, 'sha256')
	else:
		return _get_file_sha256(file_path)

def _parse_file_id(file_id):
	'''
	file_id contains one essential part and two optional parts

	file_id: name[@source][#processor]

	examples:

	.. code-block:: text

		file_id=https://XXX/            name=https://XXX/   source=None      processor=Default
		file_id=MSCOCO@tsinghua#Glove   name=MSCOCO         source=tsinghua  processor=GloveProcessor
	'''
	# There may be # in name, so we process file_id reversely.
	# TODO: what if there is # in name?
	source, processor = None, None
	name = file_id[::-1]
	if '#' in file_id:
		processor, name = name.split('#', 1)
		processor = processor[::-1]
	if '@' in file_id:
		source, name = name.split('@', 1)
		source = source[::-1]
	name = name[::-1]
	return name, source, processor

def _get_resource(file_id, cache_dir=None, config_dir=None, download=True):
	'''Get the resource with the given name.
	If not cached, download it using the URL stored in config file.
	If cached, check the hashtag.
	'''
	cache_dir = cache_dir or CACHE_DIR
	config_dir = config_dir or CONFIG_DIR
	os.makedirs(cache_dir, exist_ok=True)

	res_name, src_name, res_type = _parse_file_id(file_id)
	config = _get_config(res_name, config_dir)

	src_name = src_name or 'default'
	res_type = res_type or config.get('type', 'Default')

	resource_processor_class = ResourceProcessor.load_class(res_type + 'ResourceProcessor')
	if resource_processor_class is None:
		raise RuntimeError("No resources type named %sResourcePreprocessor" % res_type)
	resource_processor = resource_processor_class(cache_dir, config_dir)

	if src_name not in config['link']:
		raise ValueError("source {} wrong".format(src_name))
	url = config['link'][src_name]
	cache_path = os.path.join(cache_dir, _url_to_filename(res_name))
	meta_path = os.path.join(cache_dir, _url_to_filename(res_name) + '.json')
	if download:
		LOGGER.info('downloading resources')
		LOGGER.info('name: %s', res_name)
		LOGGER.info('source: %s', src_name)
		LOGGER.info('url: %s', url)
		LOGGER.info('processor: %s', res_type)

	if not os.path.exists(meta_path):
		if not download:
			#print("The resource does not exist, set download as True to download automatically")
			return None
		with tempfile.NamedTemporaryFile()  as temp_file:
			_http_get(url, temp_file)
			temp_file.flush() # flush to avoid truncation
			temp_file.seek(0) # shutil.copyfileobj() starts at the current position
			with open(cache_path, 'wb') as cache_file:
				shutil.copyfileobj(temp_file, cache_file)

		# filename hash for search, content hash for validation
		content_hash = _get_file_sha256(cache_path)

		cache_path = resource_processor.preprocess(cache_path)

		if content_hash == config['hashtag']:
			meta = {'hashtag': content_hash, 'local_path': cache_path}
			with open(meta_path, 'w', encoding='utf-8') as meta_file:
				json.dump(meta, meta_file)
		else:
			print("bad hashtag {}, correct is {}".format(content_hash, config['hashtag']))
			raise ValueError("bad hashtag of {}".format(res_name))

	else:
		with open(meta_path, 'r', encoding='utf-8') as meta_file:
			meta = json.load(meta_file)
			cache_path = meta['local_path']
			content_hash = meta['hashtag']
		if download:
			LOGGER.info('{} exists in cache'.format(res_name))
		if content_hash != config['hashtag']:
			raise ValueError("bad hashtag of {}, name conflication or mismatched content. \
							meta path {}. cache path {}".format(res_name, meta_path, cache_path))

	cache_path = resource_processor.postprocess(cache_path)
	if download:
		LOGGER.info('resource cached at %s', cache_path)
	return cache_path

def _download_data(url, cache_dir=None, config_dir=None):
	r'''If not cached, download the resource using url.
	'''
	cache_dir = cache_dir or CACHE_DIR
	config_dir = config_dir or CONFIG_DIR
	os.makedirs(cache_dir, exist_ok=True)

	url, _, res_type = _parse_file_id(url)
	res_type = res_type or 'Default'
	LOGGER.info('url: %s', url)
	LOGGER.info('processor type: %s', res_type)

	resource_processor = ResourceProcessor.load_class(res_type + 'ResourceProcessor')(cache_dir=cache_dir,config_dir=config_dir)
	cache_path = os.path.join(cache_dir, _url_to_filename(url))
	meta_path = os.path.join(cache_dir, _url_to_filename(url) + '.json')

	if not os.path.exists(meta_path):
		with tempfile.NamedTemporaryFile() as temp_file:
			_http_get(url, temp_file)
			temp_file.flush() # flush to avoid truncation
			temp_file.seek(0) # shutil.copyfileobj() starts at the current position

			with open(cache_path, 'wb') as cache_file:
				shutil.copyfileobj(temp_file, cache_file)
			# filename hash for search, content hash for validation
			content_hash = _get_file_sha256(cache_path)

			cache_path = resource_processor.preprocess(cache_path)

			meta = {'local_path': cache_path, 'hashtag': content_hash}
			with open(meta_path, 'w', encoding='utf-8') as meta_file:
				json.dump(meta, meta_file)
	else:
		with open(meta_path, 'r', encoding='utf-8') as meta_file:
			meta = json.load(meta_file)
			cache_path = meta['local_path']
	cache_path = resource_processor.postprocess(cache_path)
	LOGGER.info('resource cached at %s', cache_path)
	return cache_path

def _load_local_data(local_path, cache_dir=None, config_dir=None):
	'''Import temporary resources from local'''
	cache_dir = cache_dir or CACHE_DIR
	config_dir = config_dir or CONFIG_DIR
	local_path, _, res_type = _parse_file_id(local_path)
	res_type = res_type or 'Default'
	LOGGER.info('local path: %s', local_path)
	LOGGER.info('processor type: %s', res_type)

	resource_processor_class = ResourceProcessor.load_class(res_type + 'ResourceProcessor')
	if resource_processor_class is None:
		raise RuntimeError("No resources type named %sResourcePreprocessor" % res_type)
	resource_processor = resource_processor_class(cache_dir=cache_dir, config_dir=config_dir)

	if local_path.endswith(".zip"):
		local_path = resource_processor.preprocess(local_path)
	return resource_processor.postprocess(local_path)


def get_resource_file_path(file_id, cache_dir=None, config_dir=None, download=True):
	'''Get file_path of resource of all types
	'''
	cache_dir = cache_dir or CACHE_DIR
	config_dir = config_dir or CONFIG_DIR
	if file_id.startswith('resources://'):
		res_id = file_id[12:]
		return _get_resource(res_id, cache_dir, config_dir, download)
	elif file_id.startswith('http://') or file_id.startswith('https://'):
		url = file_id
		return _download_data(url, cache_dir, config_dir)
	else:
		local_path = file_id
		return _load_local_data(local_path, cache_dir, config_dir)

def import_local_resources(file_id, local_path, cache_dir=None, \
	config_dir=None, ignore_exist_error=False):
	'''Import benchmark from local, if hashtag checked, save to cache.'''
	cache_dir = cache_dir or CACHE_DIR
	config_dir = config_dir or CONFIG_DIR
	os.makedirs(cache_dir, exist_ok=True)

	if not file_id.startswith('resources://'):
		raise ValueError("file_id must startswith \'resources://\'")

	res_name, _, _ = _parse_file_id(file_id[12:])
	config = _get_config(res_name, config_dir)

	meta_path = os.path.join(cache_dir, _url_to_filename(res_name)) + '.json'
	if os.path.exists(meta_path):
		if ignore_exist_error:
			return
		raise ValueError("resources existed. If you want to delete the existing resources. \
			Use `rm %s`." % meta_path)

	local_hashtag = _get_hashtag(local_path)
	if local_hashtag == config['hashtag']:
		cache_path = os.path.join(cache_dir, _url_to_filename(res_name))
		with open(cache_path, 'wb') as cache_file:
			shutil.copyfileobj(open(local_path, 'rb'), cache_file)

		res_type = config.get('type', 'Default')
		resource_processor_class = ResourceProcessor.load_class(res_type + 'ResourceProcessor')
		if resource_processor_class is None:
			raise RuntimeError("No resources type named %sResourcePreprocessor" % res_type)
		resource_processor = resource_processor_class(cache_dir, config_dir)

		cache_path = resource_processor.preprocess(cache_path)
		meta = {'local_path': cache_path, 'hashtag': local_hashtag}

		with open(meta_path, 'w', encoding='utf-8') as meta_file:
			json.dump(meta, meta_file)

		cache_path = resource_processor.postprocess(cache_path)
		LOGGER.info('resource cached at %s', cache_path)
		return cache_path
	else:
		raise ValueError("bad hashtag of {}".format(res_name))

def load_file_from_url(url, force=False, cache_dir=None):
	'''See cotk.downloader.load_file_from_url.
	'''
	cache_dir = cache_dir or CACHE_DIR

	parts = urlparse(url)
	filename = os.path.basename(parts.path)

	cache_dir = os.path.join(cache_dir, 'files')

	if os.path.exists(cache_dir) and force:
		shutil.rmtree(cache_dir)
		# raise ValueError("model existed. If you want to delete the existing model. \
		# 	Use `rm %s`." % cache_path)

	os.makedirs(cache_dir, exist_ok=True)
	cache_path = os.path.join(cache_dir, filename)
	if os.path.exists(cache_path):
		return cache_path

	with tempfile.NamedTemporaryFile() as temp_file:
		_http_get(url, temp_file)
		temp_file.flush() # flush to avoid truncation
		temp_file.seek(0) # shutil.copyfileobj() starts at the current position

		with open(cache_path, 'wb') as cache_file:
			shutil.copyfileobj(temp_file, cache_file)

	LOGGER.info('model cached at %s', cache_path)
	return cache_path

def get_resource_list():
	'''get the name list of all resources'''
	files = os.listdir(CONFIG_DIR)
	files.sort()
	resources = []
	for file in files:
		resources.append(file[:file.rfind('.')])

	return resources
