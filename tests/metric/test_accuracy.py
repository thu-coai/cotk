import random
import numpy as np
import json
from metric_base import *
from cotk.metric import AccuracyMetric

def setup_module():
	random.seed(0)
	np.random.seed(0)

class TestAccuracyMetric:
	def test_version(self):
		version_test(AccuracyMetric, dataloader=None)
