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
from cotk._utils import file_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)

DASHBOARD_URL = os.getenv("COTK_DASHBOARD_URL", "") #TODO: add a online dash board url
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
It should be public.".format( \
			git_commit, git_repo, git_user \
		))

def run_model(entry, args):
	'''Run the model and record the info of library'''
	# before run model
	# cotk recorder start
	cotk.start_recorder()
	model = importlib.import_module(entry)

	try:
		model.run(args)
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
	parser.add_argument('--result', type=str, default="result.json", \
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
	upload_id = upload_report(cargs.result, cargs.entry, cargs.args, \
		git_user, git_repo, git_commit, \
		cotk_record_information, token)
	LOGGER.info("Upload complete. Check %s for your report.", SHOW_URL % upload_id)

def download(args):
	'''Entrance of download'''
	parser = argparse.ArgumentParser(prog="cotk download", \
		description='Download model information from cotk model dashboard.')
	parser.add_argument("--zip_url", type=str, default="", \
						help="Zipfile url to download from online git repo")
	parser.add_argument("--id", type=int, default=-1, help="query id")
	parser.add_argument("--result", type=str, default="result.json", \
						help="Path to dump query result.")
	parser.add_argument("--model_url", required=True, type=str, help="Model url")
	cargs = parser.parse_args(args)

	try:
		model_dir = file_utils.load_model_from_url(cargs.model_url)
	except ValueError as err:
		LOGGER.warning(str(err))
		model_dir = re.match(r'.*`rm (\S+)`\.', str(err)).groups()[0]

	if cargs.zip_url:
		match_res = re.match(r'https?://github.com/(\S+)/(\S+)/archive/(\S+).zip', cargs.zip_url)
		if match_res:
			git_user, git_repo, git_commit = match_res.groups()
			assert_commit_exist(git_user, git_repo, git_commit)
			extract_dir = '{}_{}'.format(git_repo, git_commit)
			code_dir = clone_codes_from_commit(git_user, git_repo, git_commit, extract_dir)
			LOGGER.info("Codes from {}/{}/{} fetched.".format(git_commit, git_repo, git_user))
			result_path = "{}/result.json".format(code_dir)
			if not os.path.isfile(result_path):
				raise ValueError("Result file ({}) is not found.".format(result_path))
			try:
				info = json.load(open(result_path, "r"))
			except json.JSONDecodeError as err:
				raise json.JSONDecodeError("{} is not a valid json. {}".format(result_path, err.msg), \
										   err.doc, err.pos)
			undefined_keys = []
			for key in ['working_dir', 'entry', 'args']:
				if key not in info:
					undefined_keys.append(key)
			if undefined_keys:
				raise RuntimeError("Undefined keys in `result.json`: {}".format(", ".format(undefined_keys)))
		else:
			raise RuntimeError("Invalid zip url.")
	else:
		LOGGER.info("Collecting info from id %d...", cargs.id)
		info = get_result_from_id(cargs.id)
		json.dump(info, open(cargs.result, "w"))
		LOGGER.info("Info from id %d saved to %s.", cargs.id, cargs.result)
		extract_dir = "{}".format(cargs.id)

		code_dir = clone_codes_from_commit(info['git_user'], info['git_repo'], \
												  info['git_commit'], extract_dir)
		LOGGER.info("Codes from id %d fetched.")

	cmd = "cd {}/{} && python {}.py".format(code_dir, info['working_dir'], info['entry'])
	if not info['args']:
		info['args'] = []
	else:
		if not isinstance(info['args'], list):
			raise RuntimeError("`args` in `result.json` should be of type `list`.")
	try:
		idx = info['args'].index('--model_dir')
		info['args'] = info['args'][:idx] + ['--model_dir', model_dir] + info['args'][idx + 2:]
	except:
		info['args'] = ['--model_dir', model_dir] + info['args']
	cmd += " {}".format(" ".join(info['args']))
	with open("{}/run_model.sh".format(extract_dir), "w") as file:
		file.write(cmd)
	LOGGER.info("Model running cmd written in {}".format("run_model.sh"))
	print("Model running cmd: \t{}".format(cmd))
	os.system("bash {}/run_model.sh".format(extract_dir))
	print(json.load(open("{}/result.json".format(code_dir), "r")))

def get_result_from_id(query_id):
	'''Query uploaded info from id'''
	query = requests.get(QUERY_URL % query_id)
	if not query.ok:
		raise RuntimeError("Cannot fetch result from id %d" % query_id)
	else:
		return json.loads(query.text)

def clone_codes_from_commit(git_user, git_repo, git_commit, extract_dirname):
	'''Download codes from commit'''
	url = "https://github.com/{}/{}/archive/{}.zip".format(git_user, git_repo, git_commit)
	subprocess.run(['rm', '-rf', "{}.zip".format(git_commit)], stdout=PIPE, stderr=PIPE)
	download_res = subprocess.run(["wget", url], stdout=PIPE, stderr=PIPE)
	if re.search(r'ERROR 404: Not Found', download_res.stderr.decode()):
		raise RuntimeError("Commit {} does not exist on github:{}/{}.".format(git_commit, \
																	git_repo, git_user))
	subprocess.run(['rm', '-rf', extract_dirname], stdout=PIPE, stderr=PIPE)
	unzip_res = subprocess.run(["unzip", "-o", "{}.zip".format(git_commit), \
								"-d", extract_dirname], \
							   stdout=PIPE, stderr=PIPE)
	err = unzip_res.stderr.decode().strip()
	if err:
		raise RuntimeError("Fail to unzip file {}.zip.\nError:\n{}".format(git_commit, err))
	relative_code_dir = re.search(r'.*creating: (\S+)\n.*', unzip_res.stdout.decode()).groups()[0][:-1]
	pwd = subprocess.run(['pwd'], stdout=PIPE, stderr=PIPE)
	code_dir = "{}/{}".format(pwd.stdout.decode().strip(), relative_code_dir)
	return code_dir

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

def import_local_resources(args):
	'''Entrance of importing local resources'''
	parser = argparse.ArgumentParser(prog="cotk import", \
		description="Import local resources")
	parser.add_argument("--file_id", type=str, help="Name of resource")
	parser.add_argument("--file_path", type=str, help="Path to resource")
	cargs = parser.parse_args(args)

	if cargs.file_id and cargs.file_path:
		file_utils.import_local_resources(cargs.file_id, cargs.file_path)
	elif cargs.file_id or cargs.file_path:
		raise RuntimeError("You should specify both the name and the path to the resource.")
	LOGGER.info("Successfully import local resource {}.".format(cargs.file_id))

def dispatch(sub_entrance, args):
	'''Dispatcher of sub-entrances'''
	if sub_entrance == 'report':
		report(args)
	elif sub_entrance == 'download':
		download(args)
	elif sub_entrance == 'import':
		import_local_resources(args)
	elif sub_entrance == 'config':
		config(args)

def main():
	'''Entry of command line'''
	sys.path.append(".")
	if len(sys.argv) > 2 and sys.argv[1] == "debug":
		dispatch(sys.argv[2], sys.argv[3:])
	elif len(sys.argv) > 1:
		try:
			dispatch(sys.argv[1], sys.argv[2:])
		except Exception as err: #pylint: disable=broad-except
			print("%s: %s" % (type(err).__name__, err))

if __name__ == "__main__":
	dispatch(sys.argv[1], sys.argv[2:])
