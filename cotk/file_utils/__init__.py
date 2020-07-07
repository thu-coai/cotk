'''Utils for downloading and processing file.
'''

from .file_utils import get_resource_file_path, import_local_resources, load_file_from_url, get_resource_list
from .resource_processor import ResourceProcessor, DefaultResourceProcessor, ZipResourceProcessor, \
    BaseResourceProcessor, MSCOCOResourceProcessor, OpenSubtitlesResourceProcessor, UbuntuResourceProcessor, \
    SwitchboardCorpusResourceProcessor, SSTResourceProcessor, GloveResourceProcessor, \
    Glove50dResourceProcessor, Glove100dResourceProcessor, Glove200dResourceProcessor, \
    Glove300dResourceProcessor

__all__ = ['get_resource_file_path', 'import_local_resources', 'load_file_from_url', 'get_resource_list']
