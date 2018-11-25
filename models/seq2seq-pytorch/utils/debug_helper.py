import logging
import ptvsd

def debug(secret='my_secret'):
	ptvsd.enable_attach(secret)
	# tcp://my_secret@gpu-kappa:5678
	logging.info("wait debug")
	ptvsd.wait_for_attach()