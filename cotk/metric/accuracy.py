r"""
Containing some classes and functions about accuracy evaluating results of models.
"""
import numpy as np
from .._utils import hooks
from .metric import MetricBase

class AccuracyMetric(MetricBase):
	'''Metric for calculating accuracy.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.LABEL_KEY_ARGUMENTS}
		{MetricBase.PREDICTION_KEY_ARGUMENTS}


	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> label_key = 'label'
		>>> prediction_key = "prediction"
		>>> metric = cotk.metric.AccuracyMetric(dl,
		...	    label_key=label_key,
		...	    prediction_key=prediction_key)
		>>> data = {
		...	    label_key: [1,2,2,1],
		...	    prediction_key: [1,2,1,2]
		... }
		>>> metric.forword(data)
		>>> metric.close()
		{'accuracy': 0.5,
		'accuracy hashvalue': '913ba1d873921e28c4f8964cd1683d4301e3712a351672b5129f3fc3fac53852'}

	'''

	_name = 'AccuracyMetric'
	_version = 1

	@hooks.hook_metric
	def __init__(self, dataloader,\
			label_key="label", prediction_key="prediction"):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.label_key = label_key
		self.prediction_key = prediction_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.LABEL_ARGUMENTS}
				{MetricBase.PREDICTION_ARGUMENTS}

				Here is an example for data:
					>>> data = {
					...		label_key: [1,2,2,1],
					...		prediction_key: [1,2,1,2]
					... }
		'''
		super().forward(data)
		self.hyps.extend(data[self.prediction_key])
		self.refs.extend(data[self.label_key])
		if len(data[self.prediction_key]) != len(data[self.label_key]):
			raise ValueError("Batch num is not matched.")

		self._hash_relevant_data(data[self.label_key])

	@hooks.hook_metric_close
	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **accuracy**: accuracy value.
			* **accuracy hashvalue**: hash value for accuracy metric, same hash value stands
			  for same evaluation settings.
		'''
		result = super().close()
		if (not self.hyps) or (not self.refs):
			raise RuntimeError("The metric has not been forwarded data correctly.")
		result.update({"accuracy": \
			np.mean(np.array(self.refs) == np.array(self.hyps)), \
			"accuracy hashvalue": self._hashvalue()})
		return result
