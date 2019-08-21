r"""
``cotk._utils`` often is used by internal lib. The users should
not find api here.
"""

from .file_utils import get_resource_file_path, import_local_resources
from .resource_processor import ResourceProcessor, DefaultResourceProcessor
from ._utils import trim_before_target
from .hooks import start_recorder, close_recorder

__all__ = ['ResourceProcessor', 'DefaultResourceProcessor', 'get_resource_file_path', \
        'import_local_resources']
