"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import logging
import json
import tempfile
import shutil
import hashlib
from pathlib import Path

from tqdm import tqdm
from checksumdir import dirhash

import requests

from .resource_processor import DefaultResourceProcessor

LOGGER = logging.getLogger(__name__)
CACHE_DIR = os.path.join(Path.home(), '.contk_cache')
CONFIG_DIR = './contk/resource_config'

def get_config(res_name, config_dir):
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
	for chunk in req.iter_content(chunk_size=1024):
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


def get_resource(res_name, cache_dir=CACHE_DIR, config_dir=CONFIG_DIR):
	'''Get the resource with the given name.
	If not cached, download it using the URL stored in config file.
	If cached, check the hashtag.
	'''
	os.makedirs(cache_dir, exist_ok=True)

	config = get_config(res_name, config_dir)

	cache_path = os.path.join(cache_dir, res_name)
	meta_path = cache_path + '.json'

	if not os.path.exists(meta_path):
		with tempfile.NamedTemporaryFile() as temp_file:
			url = config['link']
			LOGGER.info("%s not found in cache, downloading to %s", url, \
				temp_file.name)

			http_get(url, temp_file)

			# flush to avoid truncation
			temp_file.flush()
			# shutil.copyfileobj() starts at the current position
			temp_file.seek(0)

			LOGGER.info("copying %s to cache at %s", temp_file.name, cache_path)
			with open(cache_path, 'wb') as cache_file:
				shutil.copyfileobj(temp_file, cache_file)
			LOGGER.info("removing temp file %s", temp_file.name)

			LOGGER.info("preprocessing ...")
			cache_path = DefaultResourceProcessor().preprocess(cache_path)

			cache_hashtag = get_hashtag(cache_path)

			if cache_hashtag == config['hashtag']:
				LOGGER.info("hashtag %s checked", cache_hashtag)

				LOGGER.info("creating metadata file for %s", cache_path)
				meta = {'local_path': cache_path}
				with open(meta_path, 'w') as meta_file:
					json.dump(meta, meta_file)
			else:
				LOGGER.info("local hashtag %s differs with standard %s", \
					cache_hashtag, config['hashtag'])
				raise ValueError("bad hashtag of {}".format(res_name))
	else:
		with open(meta_path, 'r') as meta_file:
			meta = json.load(meta_file)
			cache_path = meta['local_path']

		cache_hashtag = get_hashtag(cache_path)

		if cache_hashtag == config['hashtag']:
			LOGGER.info("hashtag %s checked", cache_hashtag)
		else:
			LOGGER.info("local hashtag %s differs with %s", \
				cache_hashtag, config['hashtag'])
			raise ValueError("bad hashtag of {}".format(res_name))

	return DefaultResourceProcessor().postprocess(cache_path)


def import_local_benchmark(res_name, local_path, cache_dir=CACHE_DIR, \
	config_dir=CONFIG_DIR):
	'''Import benchmark from local, if hashtag checked, save to cache.'''
	config = get_config(res_name, config_dir)

	local_hashtag = get_hashtag(local_path)
	if local_hashtag == config['hashtag']:
		LOGGER.info("hashtag %s checked", local_hashtag)

		LOGGER.info("creating metadata file for %s", local_path)
		meta = {'local_path': local_path}
		meta_path = os.path.join(cache_dir, res_name) + '.json'
		with open(meta_path, 'w') as meta_file:
			json.dump(meta, meta_file)

		return DefaultResourceProcessor().postprocess(local_path)
	else:
		LOGGER.info("local hashtag %s differs with standard %s", local_hashtag, \
			config['hashtag'])
		raise ValueError("bad hashtag of {}".format(res_name))


def import_local_resource(local_path):
	'''Import temporary resources from local'''
	return DefaultResourceProcessor().postprocess(local_path)
