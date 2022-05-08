"""Top-level package for mbgdml."""

from . import _version
__version__ = _version.get_versions()['version']

from mbgdml import data
from mbgdml import parse
from mbgdml import train
from mbgdml import utils
from mbgdml import criteria
from mbgdml import analysis
