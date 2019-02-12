r"""
`contk._utils` provides classes and functions downloading and
importing datasets and wordvectors automatically.
"""

from .file_utils import *
from .resource_processor import ResourceProcessor, DefaultResourceProcessor
from ._utils import *

__all__ = ['ResourceProcessor', 'DefaultResourceProcessor']
