"""Top-level package for mbgdml."""

from . import _version
__version__ = _version.get_versions()['version']

import logging
from .logger import GDMLLogger
logging.setLoggerClass(GDMLLogger)