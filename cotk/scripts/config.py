'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import json
import argparse
from .._utils import file_utils
from . import main

def save_token(token):
	'''Save token locally'''
	json.dump({'token': token}, open(main.CONFIG_FILE, "w"))
	main.LOGGER.info("Save your configuration locally at {}".format(main.CONFIG_FILE))

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
