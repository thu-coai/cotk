'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import json
import argparse
from .._utils import file_utils
from . import main

def load_config():
	try:
		config_ = json.load(open(main.CONFIG_FILE, 'r', encoding='utf-8'))
	except (FileNotFoundError, json.JSONDecodeError):
		config_ = {}
	return config_


def config_set(variable, value):
	config = load_config()
	config[variable] = value
	json.dump(config, open(main.CONFIG_FILE, 'w', encoding='utf-8'))
	main.LOGGER.info("Save your configuration locally at {}".format(main.CONFIG_FILE))


def config_load(variable):
	config = load_config()
	return config.get(variable, None)

def config(args):
	'''Entrance of configuration'''
	parser = argparse.ArgumentParser(prog="cotk config", \
		description='Configuration (e.g. token)')
	parser.add_argument("action", type=str, choices=["set", "show"])
	parser.add_argument("variable", type=str)
	parser.add_argument("value", type=str, nargs="*")
	cargs = parser.parse_args(args)

	if cargs.action == "set":
		config_set(cargs.variable, " ".join(cargs.value))
		main.LOGGER.info("%s = %s", cargs.variable, " ".join(cargs.value))
	elif cargs.action == "show":
		value = config_load(cargs.variable)
		main.LOGGER.info("%s = %s", cargs.variable, value)
	else:
		raise RuntimeError("Unkown action.")
