'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import os
import os.path
import json
import sys
import argparse
import importlib
import traceback

import requests
from . import _utils, main
from .._utils import start_recorder, close_recorder

LOGGER = main.LOGGER
REPORT_URL = main.REPORT_URL
SHOW_URL = main.SHOW_URL
QUERY_URL = main.QUERY_URL
BACKUP_FILE = '.cotk_upload_backup'

def run_model(entry, args):
	'''Run the model and record the info of library'''
	# before run model
	# cotk recorder start
	import cotk
	start_recorder()
	sys.path.insert(0, os.getcwd())
	model = importlib.import_module(entry)

	if args is None:
		args = []

	try:
		model.run(*args)
	except Exception as _: #pylint: disable=broad-except
		traceback.print_exc()
		sys.exit(1)

	# after run model
	# cotk recorder end
	return close_recorder()

def read_and_validate_result(result_path):
	if not os.path.isfile(result_path):
		raise ValueError("Result file ({}) is not found.".format(result_path))
	try:
		result = json.load(open(result_path, "r"))
	except json.JSONDecodeError as err:
		raise json.JSONDecodeError("{} is not a valid json. {}".format(result_path, err.msg),\
				err.doc, err.pos)
	return result


def upload_report(result_path, entry, args, working_dir, \
	git_user, git_repo, git_commit, \
	cotk_record_information, token):
	'''Upload report to dashboard. Return id of the new record.'''
	# check result file existence
	# get git link
	result = read_and_validate_result(result_path)

	upload_information = { \
		"entry": entry, \
		"args": args, \
		"working_dir": working_dir, \
		"git_user": git_user, \
		"git_repo": git_repo, \
		"git_commit": git_commit, \
		"record_information": cotk_record_information, \
		"result": result \
	}
	#LOGGER.info("Save your report locally at {}".format(BACKUP_FILE))
	#json.dump(upload_information, open(BACKUP_FILE, 'w'))
	LOGGER.info("Uploading your report...")
	res = requests.post(REPORT_URL, {"data": json.dumps(upload_information), "token": token})
	res = json.loads(res.text)
	if res['code'] != "ok":
		raise RuntimeError("upload error. %s" % json.loads(res['err']))
	return res['id']

def get_local_token():
	'''Read locally-saved token'''
	if os.path.exists(main.CONFIG_FILE):
		return json.load(open(main.CONFIG_FILE, 'r'))['token']
	else:
		raise RuntimeError("Please config your token, \n" + \
						   "either by setting it temporarily in cotk\n" + \
						   "or by calling `cotk config`")

def verify_token_online(token):
	#TODO: wait for api
	return True

def run(args):
	'''Entrance of run'''
	parser = argparse.ArgumentParser(prog="cotk run", \
		description='Run model and report performance to cotk dashboard.\
**More args can added at end of the command to pass to your model**.')
	parser.add_argument('--token', type=str, default=None, \
		help='Use a temporary token. (Specified your accounts on dashboard.)')
	parser.add_argument('--result', type=str, default="result.json", \
		help='Path to result file that your model generated. Default: result.json')
	parser.add_argument("--only-run", action="store_true", \
		help="Just run my model, save running information to local folder but do not upload anything.")
	parser.add_argument('--only-upload', action="store_true", \
		help="Don't run my model, just upload the existing result.")
	parser.add_argument('--entry', type=str, default="main", nargs='?',\
		help="Entry file of your model, suffix name '.py' should not be included. \
			  Default: main")
	#parser.add_argument('args', nargs='*', help="args passed to your model")

	cargs, extra_args = parser.parse_known_args(args)

	if not cargs.only_run:
		if cargs.token:
			token = cargs.token
		else:
			token = get_local_token()
		verify_token_online(token)

		_utils.assert_repo_exist()
		if not _utils.check_repo_clean():
			raise RuntimeError("Your changes of code hasn't been committed. Use \"git status\" \
to check your changes.")

		git_user, git_repo = _utils.get_repo_remote()
		git_commit = _utils.get_repo_commit()
		#_utils.assert_commit_exist(git_user, git_repo, git_commit)
		LOGGER.info("git information detected.")
		LOGGER.info("user: %s, repo: %s, commit sha1: %s", git_user, git_repo, git_commit)

	try:
		_utils.assert_repo_exist()
		root_path = _utils.get_repo_root_path()
		config_file = os.path.join(root_path, ".model_config.json")
		in_git = True
	except RuntimeError:
		config_file = ".model_config.json"
		in_git = False


	if not cargs.only_upload:
		LOGGER.info("Running your model at '%s' with arguments: %s.", cargs.entry, extra_args)
		cotk_record_information = run_model(cargs.entry, extra_args)
		LOGGER.info("Your model has exited.")

		if in_git:
			working_dir = _utils.get_repo_workingdir()
			data = {"entry": cargs.entry, "args": extra_args,\
			"cotk_record_information": cotk_record_information, "working_dir": working_dir}
		else:
			data = {"entry": cargs.entry, "args": extra_args,\
			"cotk_record_information": cotk_record_information}

		json.dump(data, open(config_file, 'w'))
		LOGGER.info("Runtime information has dumped into %s.", config_file)

		validate_result(cargs.result)
	else:

		if not os.path.isfile(config_file):
			raise RuntimeError(".model_config.json not found. It seems you have not run your \
model with cotk. You can't use \"only-upload\" before \"only-run\".")
		data = json.load(open(config_file))

	if not cargs.only_run:
		LOGGER.info("Collecting info for upload...")
		upload_id = upload_report(cargs.result, data["entry"], data["args"], data.get("working_dir", ''),\
			git_user, git_repo, git_commit, \
			data["cotk_record_information"], token)
		LOGGER.info("Upload complete. Check %s for your report.", SHOW_URL % upload_id)
