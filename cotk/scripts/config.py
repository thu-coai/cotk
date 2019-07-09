'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import json
import argparse
from cotk._utils import file_utils
from cotk.scripts import entry

def save_token(token):
	'''Save token locally'''
	json.dump({'token': token}, open(entry.CONFIG_FILE, "w"))
	entry.LOGGER.info("Save your configuration locally at {}".format(entry.CONFIG_FILE))

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


def import_local_resources(args):
	'''Entrance of importing local resources'''
	parser = argparse.ArgumentParser(prog="cotk import", \
		description="Import local resources")
	parser.add_argument("--file_id", type=str, required=True, help="Name of resource")
	parser.add_argument("--file_path", type=str, required=True, help="Path to resource")
	cargs = parser.parse_args(args)

	file_utils.import_local_resources(cargs.file_id, cargs.file_path)
	entry.LOGGER.info("Successfully import local resource {}.".format(cargs.file_id))
