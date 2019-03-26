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

from tqdm import tqdm
from checksumdir import dirhash

import requests

from .resource_processor import ResourceProcessor

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)
CACHE_DIR = os.path.join(str(Path.home()), '.cotk_cache')
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../resource_config')

def url_to_filename(url):
	r'''Convert the url to sha256 as filename
	'''
	url_bytes = url.encode('utf-8')
	url_hash = hashlib.sha256(url_bytes)
	filename = url_hash.hexdigest()

	return filename


def get_config(res_name, config_dir=CONFIG_DIR):
	'''Get config(dict) by the name of resource'''
	config_path = os.path.join(config_dir, res_name + '.json')
	if not os.path.exists(config_path):
		raise FileNotFoundError("file {} not found".format(config_path))
	with open(config_path, 'r') as config_file:
		config = json.load(config_file)

	return config


def http_get(url, temp_file):
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


def get_file_sha256(file_path):
	'''Get sha256 of given file'''
	hash_sha256 = hashlib.sha256()
	with open(file_path, "rb") as fin:
		for chunk in iter(lambda: fin.read(4096), b""):
			hash_sha256.update(chunk)
	return hash_sha256.hexdigest()


def get_hashtag(file_path):
	'''Get sha256 of given directory or file'''
	if os.path.isdir(file_path):
		return dirhash(file_path, 'sha256')
	else:
		return get_file_sha256(file_path)


def get_resource(res_name, res_type, cache_dir=CACHE_DIR, config_dir=CONFIG_DIR):
	'''Get the resource with the given name.
	If not cached, download it using the URL stored in config file.
	If cached, check the hashtag.
	'''
	os.makedirs(cache_dir, exist_ok=True)

	if '~' in res_name:
		res_name, src_name = res_name.split('~', 1)
	else:
		src_name = 'github'

	config = get_config(res_name, config_dir)
	if config['type'] != res_type:
		raise ValueError("res_type {} differs with res_type {}".format(res_type, config['type']))

	resource_processor = ResourceProcessor.load_class(res_type + 'ResourceProcessor')()
	if resource_processor is None:
		raise RuntimeError("No resources type named %sResourcePreprocessor" % res_type)
	if src_name not in config['link']:
		raise ValueError("source {} wrong".format(src_name))
	url = config['link'][src_name]
	cache_path = os.path.join(cache_dir, url_to_filename(url))
	meta_path = os.path.join(cache_dir, url_to_filename(url) + '.json')

	if not os.path.exists(meta_path):
		with tempfile.NamedTemporaryFile() as temp_file:
			http_get(url, temp_file)
			temp_file.flush() # flush to avoid truncation
			temp_file.seek(0) # shutil.copyfileobj() starts at the current position

			with open(cache_path, 'wb') as cache_file:
				shutil.copyfileobj(temp_file, cache_file)

			cache_path = resource_processor.preprocess(cache_path)

			cache_hashtag = get_hashtag(cache_path)
			if cache_hashtag == config['hashtag']:
				meta = {'local_path': cache_path}
				with open(meta_path, 'w') as meta_file:
					json.dump(meta, meta_file)
			else:
				print("bad hashtag {}, correct is {}".format(cache_hashtag, config['hashtag']))
				raise ValueError("bad hashtag of {}".format(res_name))
	else:
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			cache_path = meta['local_path']

		cache_hashtag = get_hashtag(cache_path)
		if cache_hashtag != config['hashtag']:
			raise ValueError("bad hashtag of {}".format(res_name))

	cache_path = resource_processor.postprocess(cache_path)
	LOGGER.info('resource cached at %s', cache_path)
	return cache_path


def download_resource(url, res_type, cache_dir=CACHE_DIR):
	r'''If not cached, download the resource using url.
	'''
	os.makedirs(cache_dir, exist_ok=True)

	resource_processor = ResourceProcessor.load_class(res_type + 'ResourceProcessor')()
	cache_path = os.path.join(cache_dir, url_to_filename(url))
	meta_path = os.path.join(cache_dir, url_to_filename(url) + '.json')

	if not os.path.exists(meta_path):
		with tempfile.NamedTemporaryFile() as temp_file:
			http_get(url, temp_file)
			temp_file.flush() # flush to avoid truncation
			temp_file.seek(0) # shutil.copyfileobj() starts at the current position

			with open(cache_path, 'wb') as cache_file:
				shutil.copyfileobj(temp_file, cache_file)

			cache_path = resource_processor.preprocess(cache_path)

			meta = {'local_path': cache_path}
			with open(meta_path, 'w') as meta_file:
				json.dump(meta, meta_file)
	else:
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			cache_path = meta['local_path']
	cache_path = resource_processor.postprocess(cache_path)
	LOGGER.info('resource cached at %s', cache_path)
	return cache_path


def import_local_benchmark(res_name, local_path, cache_dir=CACHE_DIR, \
	config_dir=CONFIG_DIR):
	'''Import benchmark from local, if hashtag checked, save to cache.'''
	config = get_config(res_name, config_dir)

	local_hashtag = get_hashtag(local_path)
	if local_hashtag == config['hashtag']:

		meta = {'local_path': local_path}
		meta_path = os.path.join(cache_dir, res_name) + '.json'
		with open(meta_path, 'w') as meta_file:
			json.dump(meta, meta_file)

		return local_path
	else:
		raise ValueError("bad hashtag of {}".format(res_name))


def import_local_resource(local_path, res_type):
	'''Import temporary resources from local'''
	resource_processor = ResourceProcessor.load_class(res_type + 'ResourceProcessor')()
	return resource_processor.postprocess(local_path)


def get_resource_file_path(file_id, res_type="Default", cache_dir=CACHE_DIR, config_dir=CONFIG_DIR):
	'''Get file_path of resource of all types
	'''
	if file_id.startswith('resources://'):
		res_id = file_id[12:]
		return get_resource(res_id, res_type, cache_dir, config_dir)
	elif file_id.startswith('http://') or file_id.startswith('https://'):
		url = file_id
		return download_resource(url, res_type, cache_dir)
	else:
		local_path = file_id
		return import_local_resource(local_path, res_type)
