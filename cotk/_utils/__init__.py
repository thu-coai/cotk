r"""
`cotk._utils` provides classes and functions downloading and
importing datasets and wordvectors automatically.
"""

from .file_utils import *
from .resource_processor import ResourceProcessor, DefaultResourceProcessor
from ._utils import *
from .hooks import start_recorder, close_recorder

__all__ = ['ResourceProcessor', 'DefaultResourceProcessor']
