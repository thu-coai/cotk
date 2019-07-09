'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import logging
import os
import os.path
import json
import argparse
import re
import shutil
import zipfile

import requests
from cotk._utils import file_utils
from cotk.scripts import _utils
from cotk.scripts import entry

def get_result_from_id(query_id):
	'''Query uploaded info from id'''
	query = requests.get(entry.QUERY_URL % query_id)
	if not query.ok:
		raise RuntimeError("Cannot fetch result from id %d" % query_id)
	else:
		return json.loads(query.text)

def clone_codes_from_commit(git_user, git_repo, git_commit, extract_dirname):
	'''Download codes from commit'''
	_utils.assert_commit_exist(git_user, git_repo, git_commit)
	url = "https://github.com/{}/{}/archive/{}.zip".format(git_user, git_repo, git_commit)
	zip_filename = "{}.zip".format(git_commit)
	if os.path.exists(zip_filename):
		resp = input("{} exists. Do you want to overwrite it? (y or n)")
		if resp == 'y':
			shutil.rmtree(zip_filename)
			file_utils._http_get(url, open(zip_filename, "wb"))
	else:
		file_utils._http_get(url, open(zip_filename, "wb"))
	if not zipfile.is_zipfile(zip_filename):
		raise RuntimeError("{} is not a zip file".format(zip_filename))
	# shutil.rmtree(extract_dirname, ignore_errors=True)
	zip_file = zipfile.ZipFile(zip_filename, 'r')
	zip_file.extractall(extract_dirname)
	relative_code_dir = extract_dirname + "/" + zip_file.namelist()[0].split("/")[0]
	zip_file.close()
	code_dir = "{}/{}".format(os.getcwd(), relative_code_dir)
	return code_dir

def download(args):
	'''Entrance of download'''
	parser = argparse.ArgumentParser(prog="cotk download", \
		description='Download model and information from github and dashboard.',\
		formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("model", type=str, help=\
r"""A string indicates model path. It can be one of following:
* Id of cotk dashboard. Example: 12.
* Url of a git repo / branch / commit.
    Example: https://github.com/USER/REPO
             https://github.com/USER/REPO/tree/BRANCH
             https://github.com/USER/REPO/commit/COMMIT (full commit id should be included)
* A string specifying a git repo / branch / commit.
    Example: USER/REPO
             USER/REPO/BRANCH
             USER/REPO/COMMIT (full commit id should be included)
""")
	parser.add_argument("--result", type=str, default="dashboard_result.json", \
						help="Path to dump query result.")
	cargs = parser.parse_args(args)

	if cargs.model.isalnum():
		# download from dashboard
		board_id = int(cargs.model)
		entry.LOGGER.info("Collecting info from id %d...", board_id)
		info = get_result_from_id(board_id)
		json.dump(info, open(cargs.result, "w"))
		entry.LOGGER.info("Info from id %d saved to %s.", board_id, cargs.result)
		extract_dir = "{}".format(cargs.id)

		code_dir = clone_codes_from_commit(info['git_user'], info['git_repo'], \
												  info['git_commit'], extract_dir)
		entry.LOGGER.info("Codes from id %d fetched.")
	else:
		# download from online git repo
		patterns_2 = r'(?:https?://github\.com/)?(\w+)/(\w+)/?'
		patterns_3 = r'(?:https?://github\.com/)?(\w+)/(\w+)/(?:(?:tree|commit)/)?(\w+)/?'
		match_res = re.fullmatch(patterns_2, cargs.model)
		if match_res:
			git_user, git_repo = match_res.groups()
			git_commit = "master"
		else:
			match_res = re.fullmatch(patterns_3, cargs.model)
			if not match_res:
				raise ValueError("'%s' can't match any pattern." % cargs.model)
			git_user, git_repo, git_commit = match_res.groups()

		entry.LOGGER.info("Fetching {}/{}/{}".format(git_user, git_repo, git_commit))
		extract_dir = '{}_{}'.format(git_repo, git_commit)
		code_dir = clone_codes_from_commit(git_user, git_repo, git_commit, extract_dir)
		entry.LOGGER.info("Codes from {}/{}/{} fetched.".format(git_user, git_repo, git_commit))
		config_path = "{}/config.json".format(code_dir)
		if not os.path.isfile(config_path):
			raise FileNotFoundError("Config file ({}) is not found.".format(config_path))
		try:
			info = json.load(open(config_path, "r"))
		except json.JSONDecodeError as err:
			raise json.JSONDecodeError("{} is not a valid json. {}".format(config_path, err.msg), \
										err.doc, err.pos)
		undefined_keys = []
		for key in ['working_dir', 'entry', 'args']:
			if key not in info:
				undefined_keys.append(key)
		if undefined_keys:
			raise RuntimeError("Undefined keys in `config.json`: {}".format(", ".join(undefined_keys)))

	# cmd construction
	cmd = "cd {}/{} && cotk run --only-run --entry {}".format(code_dir, info['working_dir'], info['entry'])
	if not info['args']:
		info['args'] = []
	else:
		if not isinstance(info['args'], list):
			raise ValueError("`args` in `config.json` should be of type `list`.")

	cmd += " {}".format(" ".join(info['args']))
	with open("{}/run_model.sh".format(extract_dir), "w") as file:
		file.write(cmd)
	entry.LOGGER.info("Model running cmd written in {}".format("run_model.sh"))
	print("Model running cmd: \t{}".format(cmd))

	# run model
	# result_path = "{}/result.json".format(code_dir)
	# old_time_stamp = 0
	# if os.path.exists(result_path):
	# 	old_time_stamp = os.path.getmtime(result_path)

	# os.system("bash {}/run_model.sh".format(extract_dir))
	# if not os.path.exists(result_path) or \
	# 	os.path.getmtime('{}/result.json'.format(code_dir)) <= old_time_stamp:
	# 	raise FileNotFoundError("New result file not found.")
	# print(json.load(open("{}/result.json".format(code_dir), "r")))
