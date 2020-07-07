'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python

import argparse

from ..file_utils import get_resource_file_path, get_resource_list
from . import cli_constant as cli

def entry(args):
	'''Entrance of show resources path and whether resource is cached or not'''
	resource_names = get_resource_list()
	parser = argparse.ArgumentParser(prog="cotk resources", \
		description="check resources site and whether s specific resource cache is available")
	parser.add_argument("--show_all", action="store_true", help="Show path of all resources")
	parser.add_argument("--show_stored", action="store_true", help="Show path of all stored resource")
	parser.add_argument("--show", type=str, help="Show path of a specific resource")
	cargs = parser.parse_args(args)
	if cargs.show_all:
		cli.LOGGER.info("{:30}\t{:100}".format(
			"Resource IDs", "Cache paths"))
		for resource in resource_names:
			cache_path = get_resource_file_path("resources://"+resource, download=False)
			if cache_path is not None:
				cli.LOGGER.info("{:30}\t{:100}".format(
					resource, cache_path))
			else:
				cli.LOGGER.info("{:30}\t{:100}".format(
					resource, "Not cached"))

	elif cargs.show_stored:
		cli.LOGGER.info("{:30}\t{:100}".format(
			"Resource IDs", "Cache paths"))
		for resource in resource_names:
			cache_path = get_resource_file_path("resources://"+resource, download=False)
			if cache_path is not None:
				cli.LOGGER.info("{:30}\t{:100}".format(
					resource, cache_path))

	elif cargs.show is not None:
		if cargs.show[:12] != ("resources://"):
			raise RuntimeError('Please input a string starting with "resources://"')
		if cargs.show[12:] not in resource_names:
			raise RuntimeError("Unkown resource name {}".format(cargs.show[12:]))
		cache_path = get_resource_file_path(cargs.show, download=False)
		if cache_path is not None:
			cli.LOGGER.info("{:30}\t{:100}".format(
				"Resource IDs", "Cache paths"))
			cli.LOGGER.info("{:30}\t{:100}".format(
				cargs.show, cache_path))
		else:
			cli.LOGGER.info("resource {} is not cached.".format(cargs.show))
	else:
		raise RuntimeError("Unkown params.")
