try:  # python 3.7+
	from typing import OrderedDict
	OrderedDictType = OrderedDict
except ImportError:
	try: # python 3.6+
		from typing import MutableMapping
		OrderedDictType = MutableMapping
	except ImportError: # python 3.5+
		from typing import Any
		OrderedDictType = Any

__all__ = ["OrderedDictType"]
