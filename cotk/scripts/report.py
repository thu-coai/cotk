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
import cotk
from cotk.scripts import _utils
from cotk.scripts import entry

LOGGER = entry.LOGGER
REPORT_URL = entry.REPORT_URL
SHOW_URL = entry.SHOW_URL
QUERY_URL = entry.QUERY_URL

def run_model(entry, args):
	'''Run the model and record the info of library'''
	# before run model
	# cotk recorder start
	cotk.start_recorder()
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
	return cotk.close_recorder()

def upload_report(result_path, entry, args, \
	git_user, git_repo, git_commit, \
	cotk_record_information, token):
	'''Upload report to dashboard. Return id of the new record.'''
	# check result file existence
	# get git link
	if not os.path.isfile(result_path):
		raise ValueError("Result file ({}) is not found.".format(result_path))
	try:
		result = json.load(open(result_path, "r"))
	except json.JSONDecodeError as err:
		raise json.JSONDecodeError("{} is not a valid json. {}".format(result_path, err.msg),\
				err.doc, err.pos)

	working_dir = _utils.get_repo_workingdir()

	upload_information = { \
		"entry": entry, \
		"args": args, \
		"working_dir": working_dir, \
		"git_user": git_user, \
		"git_repo": git_repo, \
		"git_commit": git_commit, \
		"record_information": cotk_record_information, \
		"result": json.dumps(result) \
	}
	LOGGER.info("Save your report locally at {}".format(entry.BACKUP_FILE))
	json.dump(upload_information, open(entry.BACKUP_FILE, 'w'))
	LOGGER.info("Uploading your report...")
	res = requests.post(REPORT_URL, {"data": upload_information, "token": token})
	res = json.loads(res.text)
	if res['code'] != "ok":
		raise RuntimeError("upload error. %s" % json.loads(res['err']))
	return res['id']

def get_local_token():
	'''Read locally-saved token'''
	if os.path.exists(entry.CONFIG_FILE):
		return json.load(open(entry.CONFIG_FILE, 'r'))['token']
	else:
		raise RuntimeError("Please config your token, \n" + \
						   "either by setting it temporarily in cotk\n" + \
						   "or by calling `cotk config`")

def run(args):
	'''Entrance of run'''
	parser = argparse.ArgumentParser(prog="cotk run", \
		description='Run model and report performance to cotk dashboard.')
	parser.add_argument('--token', type=str, default=None)
	parser.add_argument('--result', type=str, default="result.json", \
		help='Path to result file. Default: result.json')
	parser.add_argument("--only-run", action="store_true", \
		help="Just run my model, don't collect any information or upload anything.")
	parser.add_argument('--only-upload', action="store_true", \
		help="Don't run my model, just upload the existing result. \
		(Some information will be missing and this option is not recommended.)")
	parser.add_argument('--entry', type=str, default="main", nargs='?',\
		help="Entry of your model. Default: main")
	parser.add_argument('args', nargs=argparse.REMAINDER)

	cargs = parser.parse_args(args)

	if not cargs.only_run:
		if cargs.token:
			token = cargs.token
		else:
			token = get_local_token()
		_utils.assert_repo_exist()
		git_user, git_repo = _utils.get_repo_remote()
		git_commit = _utils.get_repo_commit()
		_utils.assert_commit_exist(git_user, git_repo, git_commit)
		LOGGER.info("git information detected.")
		LOGGER.info("user: %s, repo: %s, commit sha1: %s", git_user, git_repo, git_commit)
	if cargs.only_upload:
		LOGGER.warning("Your model is not running, only upload existing result. \
Some information will be missing and it is not recommended.")
		cotk_record_information = None
	else:
		if not cargs.only_run and not _utils.check_repo_clean():
			raise RuntimeError("Your changes of code hasn't been committed. Use \"git status\" \
to check your changes.")

		LOGGER.info("Running your model at '%s' with arguments: %s.", cargs.entry, cargs.args)
		cotk_record_information = run_model(cargs.entry, cargs.args)
		LOGGER.info("Your model has exited.")

	if not cargs.only_run:
		LOGGER.info("Collecting info for update...")
		upload_id = upload_report(cargs.result, cargs.entry, cargs.args, \
			git_user, git_repo, git_commit, \
			cotk_record_information, token)
		LOGGER.info("Upload complete. Check %s for your report.", SHOW_URL % upload_id)
