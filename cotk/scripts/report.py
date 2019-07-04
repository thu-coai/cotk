'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import logging
import os
import os.path
import json
import sys
import argparse
import importlib
import subprocess
from subprocess import PIPE
import re
import traceback

import requests
import cotk

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)

DASHBOARD_URL = os.getenv("COTK_DASHBOARD_URL", None) #TODO: add a online dash board url
REPORT_URL = DASHBOARD_URL + "/upload"
SHOW_URL = DASHBOARD_URL + "/show?id=%d"
QUERY_URL = DASHBOARD_URL + "/get?id=%d"

def assert_repo_exist():
	'''Assert cwd is in a git repo.'''
	try:
		in_git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], stdout=PIPE, stderr=PIPE)
	except FileNotFoundError as _:
		raise RuntimeError("Git is not found. You must install git and \
make sure git command can be used.")

	if in_git.stdout.decode().strip() != "true":
		raise RuntimeError("You have to make a commit in your git repo first.")

def check_repo_clean():
	'''Check whether repo is clean.
	Return True if clean, False if dirty.'''
	git_diff = subprocess.run(["git", "diff", "HEAD"], stdout=PIPE, stderr=PIPE)
	if git_diff.stdout.decode():
		return False
	else:
		return True

def get_repo_workingdir():
	'''Get relative path of cwd from git repo root.'''
	git_prefix = subprocess.run(["git", "rev-parse", "--show-prefix"], stdout=PIPE, stderr=PIPE)
	return git_prefix.stdout.decode().strip()

def get_repo_remote():
	'''Get remote repo name on github'''
	git_upstream = subprocess.run(["git", "rev-parse", "--symbolic-full-name", "--abbrev-ref", \
			"@{upstream}"], stdout=PIPE, stderr=PIPE)
	err = git_upstream.stderr.decode()
	if err:
		if re.match(r"fatal: no upstream configured for branch '\s*?'", err):
			raise RuntimeError("No upstream branch, you have to set upstream branch for your repo. \
E.g. git push -u origin master. ")
		else:
			raise RuntimeError("Unkown error when getting upstream branch：%s" % err)

	upstream_out = git_upstream.stdout.decode().split('/')
	remote_name, remote_branch = upstream_out[0], upstream_out[1] #pylint: disable=unused-variable

	git_remote = subprocess.run(["git", "remote", "-v"], stdout=PIPE, stderr=PIPE)
	ssh_reg = re.search(r"%s\s+git@github.com:(\S+?)/(\S+?)\.git\s+\(push\)" % \
			remote_name, git_remote.stdout.decode())
	http_reg = re.search(r"%s\s+https?://github\.com/(\S+?)/(\S+?)\.git\s+\(push\)" % \
			remote_name, git_remote.stdout.decode())
	if ssh_reg is None and http_reg is None:
		raise RuntimeError("No remote named %s, please use 'git remote add' to identify \
your remote repo on github." % git_remote)
	if ssh_reg:
		git_user = ssh_reg.group(1)
		git_repo = ssh_reg.group(2)
	else:
		git_user = http_reg.group(1)
		git_repo = http_reg.group(2)

	return git_user, git_repo

def get_repo_commit():
	'''Return the commit sha of HEAD'''
	git_head = subprocess.run(["git", "rev-parse", "HEAD"], stdout=PIPE, stderr=PIPE)
	if git_head.stdout.decode().find("fatal: Needed a single revision") >= 0:
		raise RuntimeError("You have to make a commit in your git repo first.")
	return git_head.stdout.decode().strip()

def assert_commit_exist(git_user, git_repo, git_commit):
	'''Assert commit is available'''
	url = "https://github.com/{}/{}/archive/{}.zip".format(git_user, git_repo, git_commit)
	res = requests.head(url)
	if not res.ok:
		raise RuntimeError("Commit {} does not exist on github:{}/{}. \
Have you pushed your commit? Or make it public?".format( \
			git_commit, git_repo, git_user \
		))

def run_model(entry, args):
	'''Run the model and record the info of library'''
	# before run model
	# cotk recorder start
	cotk.start_recorder()
	model = importlib.import_module(entry)

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

	working_dir = get_repo_workingdir()

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
	LOGGER.info("Save your report locally at .cotk_upload_backup")
	json.dump(upload_information, open(".cotk_upload_backup", 'w'))
	LOGGER.info("Uploading your report...")
	res = requests.post(REPORT_URL, {"data": upload_information, "token": token})
	res = json.loads(res.text)
	if res['code'] != "ok":
		raise RuntimeError("upload error. %s" % json.loads(res['err']))
	return res['id']

def get_local_token():
	'''Read locally-saved token'''
	if os.path.exists('.cotk_config'):
		return json.load(open('.cotk_config', 'r'))['token']
	else:
		raise RuntimeError("Please config your token, \n" + \
						   "either by setting it temporarily in cotk\n" + \
						   "or by calling ^cotk config^")

def report(args):
	'''Entrance of report'''
	parser = argparse.ArgumentParser(prog="cotk report", \
		description='Report model performance to cotk model dashboard.')
	parser.add_argument('--token', type=str, default="")
	parser.add_argument('--upload_result', type=str, default="result.json", \
		help='Path to result file. Default: result.json')
	parser.add_argument('--only-upload', action="store_true", \
		help="Don't run your model, just upload the existing result. \
		(Some information will be missing and this option is not recommended.)")
	parser.add_argument('--entry', type=str, default="main", nargs='?',\
		help="Entry of your model. Default: main")
	parser.add_argument('--args', nargs=argparse.REMAINDER)

	cargs = parser.parse_args(args)

	if cargs.token:
		token = cargs.token
	else:
		token = get_local_token()
	assert_repo_exist()
	git_user, git_repo = get_repo_remote()
	git_commit = get_repo_commit()
	assert_commit_exist(git_user, git_repo, git_commit)
	LOGGER.info("git information detected.")
	LOGGER.info("user: %s, repo: %s, commit sha1: %s", git_user, git_repo, git_commit)
	if cargs.only_upload:
		LOGGER.warning("Your model is not runing, only upload existing result. \
Some information will be missing and it is not recommended.")
		cotk_record_information = None
	else:
		if not check_repo_clean():
			raise RuntimeError("Your changes of code hasn't been committed. Use \"git status\" \
to check your changes.")
		LOGGER.info("Running your model at '%s' with arguments: %s.", cargs.entry, cargs.args)
		cotk_record_information = run_model(cargs.entry, cargs.args)

	LOGGER.info("Collecting info for update...")
	upload_id = upload_report(cargs.upload_result, cargs.entry, cargs.args, \
		git_user, git_repo, git_commit, \
		cotk_record_information, token)
	LOGGER.info("Upload complete. Check %s for your report.", SHOW_URL % upload_id)

def download(args):
	'''Entrance of download'''
	parser = argparse.ArgumentParser(prog="cotk download", \
		description='Download model information from cotk model dashboard.')
	parser.add_argument("--query_id", type=int, default=-1, help="query id")
	parser.add_argument("--query_result", type=str, default="result.json", \
						help="Path to dump query result.")
	cargs = parser.parse_args(args)

	LOGGER.info("Collecting info from id %d...", cargs.query_id)
	info = get_result_from_id(cargs.query_id)
	json.dump(info, open(cargs.query_result, "w"))
	LOGGER.info("Info from id %d saved to %s.", cargs.query_id, cargs.query_result)
	clone_codes_from_commit(cargs.query_id, info['git_user'], info['git_repo'], info['git_commit'])
	LOGGER.info("Codes from id %d fetched.")

	pwd = subprocess.run(['pwd'], stdout=PIPE, stderr=PIPE)
	cmd = "cd {}/codes_from_{}/*/{} && python {}.py".format(pwd.stdout.decode().strip(), \
															cargs.query_id, \
															info['working_dir'], \
															info['entry'])
	if not info['args']:
		cmd += " {}".format(" ".join(info['args']))
	with open("run_model.sh", "w") as file:
		file.write(cmd)
	LOGGER.info("Model running cmd written in {}".format("run_model.sh"))
	print("Model running cmd: \t{}".format(cmd))

def get_result_from_id(query_id):
	'''Query uploaded info from id'''
	query = requests.get(QUERY_URL % query_id)
	if not query.ok:
		raise RuntimeError("Cannot fetch result from id %d" % query_id)
	else:
		return json.loads(query.text)

def clone_codes_from_commit(query_id, git_user, git_repo, git_commit):
	'''Download codes from commit'''
	url = "https://github.com/{}/{}/archive/{}.zip".format(git_user, git_repo, git_commit)
	download_res = subprocess.run(["wget", url], stdout=PIPE, stderr=PIPE)
	if re.search(r'ERROR 404: Not Found', download_res.stderr.decode()):
		raise RuntimeError("Commit {} does not exist on github:{}/{}.".format(git_commit, \
																	git_repo, git_user))
	unzip_res = subprocess.run(["unzip", "{}.zip".format(git_commit), \
								"-d", "codes_from_{}".format(query_id)], \
							   stdout=PIPE, stderr=PIPE)
	err = unzip_res.stderr.decode()
	if not err:
		raise RuntimeError("Fail to unzip file {}.zip.\nError:\n{}".format(git_commit, err))

def config(args):
	'''Entrance of configuration'''
	parser = argparse.ArgumentParser(prog="cotk config", \
		description='Configuration (e.g. token)')
	parser.add_argument('--token', type=str, default="")
	cargs = parser.parse_args(args)

	if cargs.token:
		save_token(cargs.token)
	else:
		raise RuntimeError("Token cannot be empty.")

def save_token(token):
	'''Save token locally'''
	json.dump({'token': token}, open(".cotk_config", "w"))
	LOGGER.info("Save your configuration locally at .cotk_config")

def main():
	'''Entry of command line'''
	sys.path.append(".")
	if len(sys.argv) > 2 and sys.argv[1] == "debug":
		if sys.argv[2] == 'report':
			report(sys.argv[3:])
		elif sys.argv[2] == 'download':
			download(sys.argv[3:])
		elif sys.argv[2] == 'config':
			config(sys.argv[3:])
	elif len(sys.argv) > 1:
		try:
			if sys.argv[1] == 'report':
				report(sys.argv[2:])
			elif sys.argv[1] == 'download':
				download(sys.argv[2:])
			elif sys.argv[1] == 'config':
				config(sys.argv[2:])
		except Exception as err: #pylint: disable=broad-except
			print("%s: %s" % (type(err).__name__, err))

if __name__ == "__main__":
	if sys.argv[1] == 'report':
		report(sys.argv[2:])
	elif sys.argv[1] == 'download':
		download(sys.argv[2:])
	elif sys.argv[1] == 'config':
		config(sys.argv[2:])
