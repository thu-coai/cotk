'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import json
import argparse
from .._utils import file_utils
from . import main

def import_local_resources(args):
	'''Entrance of importing local resources'''
	parser = argparse.ArgumentParser(prog="cotk import", \
		description="Import local resources")
	parser.add_argument("file_id", type=str, help="Name of resource")
	parser.add_argument("file_path", type=str, help="Path to resource")
	cargs = parser.parse_args(args)

	file_utils.import_local_resources(cargs.file_id, cargs.file_path)
	main.LOGGER.info("Successfully import local resource {}.".format(cargs.file_id))
