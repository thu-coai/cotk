'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import logging
import os.path
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)
FORMAT = logging.Formatter("%(levelname)s: %(message)s")
SH = logging.StreamHandler(stream=sys.stdout)
SH.setFormatter(FORMAT)
LOGGER.addHandler(SH)

CONFIG_FILE = os.path.join(str(Path.home()), '.cotk_config')
DASHBOARD_URL = os.getenv("COTK_DASHBOARD_URL", "http://coai.cs.tsinghua.edu.cn/dashboard")

def show_command():
	'''show help'''
	print(r"""usage: cotk <command> [...]

command list:
   run          Push your model result.
   download     Download a model online.
   import       Import local files to cotk cache.
   config       Settings.

You can type `cotk <command>` for details of each command.
""")

def dispatch(sub_entrance, args):
	'''Dispatcher of sub-entrances'''
	if sub_entrance == 'run':
		from . import report
		report.run(args)
	elif sub_entrance == 'download':
		from . import download
		download.download(args)
	elif sub_entrance == 'import':
		from . import import_local_resources
		import_local_resources.import_local_resources(args)
	elif sub_entrance == 'config':
		from . import config
		config.config(args)
	else:
		print("Unknown command.\n")
		show_command()

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
	else:
		show_command()

if __name__ == "__main__":
	dispatch(sys.argv[1], sys.argv[2:])
