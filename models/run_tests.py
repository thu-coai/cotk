#!/usr/bin/python3

import argparse
import os
import sys

models_path = os.path.dirname(os.path.abspath(__file__))
dirs = [f for f in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, f))]

for dirname in dirs:
	print("testing %s" % dirname)
	os.chdir(os.path.join(models_path, dirname))
	os.system("pip install -r requirements.txt")
	ret = os.system("py.test --cov=./ ./ --cov-report term-missing --cov-append --cov-config ../.coveragerc")
	ret = ret >> 8
	print("pytest return %d" % ret)
	if ret != 0 and ret != 5:
		print("test failed")
		sys.exit(ret)
sys.exit(0)
