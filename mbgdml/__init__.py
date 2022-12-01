"""Top-level package for mbgdml."""

import logging
from . import _version

__version__ = _version.get_versions()["version"]

from .logger import GDMLLogger

logging.setLoggerClass(GDMLLogger)
