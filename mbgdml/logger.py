# MIT License
#
# Copyright (c) 2020-2023, Alex M. Maldonado
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import random
import time
import numpy as np
from qcelemental import periodictable as ptable
from . import _version

__version__ = _version.get_versions()["version"]


def set_log_level(level):
    """Dynamically control the log level of mbGDML.

    Parameters
    ----------
    level : :obj:`int`
        The desired logging level.
    """
    for logger_name in logging.root.manager.loggerDict:  # pylint: disable=no-member
        if "mbgdml" in logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


def atoms_by_element(atom_list):
    r"""Converts a list of atoms identified by their atomic number to their
    elemental symbol in the same order.

    Parameters
    ----------
    atom_list : :obj:`list` [:obj:`int`]
        Atomic numbers of atoms within a structure.

    Returns
    -------
    :obj:`list` [:obj:`str`]
        Element symbols of atoms within a structure.
    """
    return [ptable.to_symbol(z) for z in atom_list]


class GDMLLogger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.formatter = logging.Formatter("%(message)s")
        self.handler = logging.StreamHandler()
        self.handler.setLevel(level)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        self.t_hashes = {}

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.log(logging.CRITICAL, msg, *args, **kwargs)

    def log_array(self, array, level=logging.INFO):
        if isinstance(array, (list, tuple)):
            array = np.array(array)
        arr_str = np.array2string(array, separator=", ")
        self.logger.log(level, arr_str)

    def log_package(self):
        title = r"""           _      ____ ____  __  __ _
 _ __ ___ | |__  / ___|  _ \|  \/  | |    
| '_ ` _ \| '_ \| |  _| | | | |\/| | |    
| | | | | | |_) | |_| | |_| | |  | | |___ 
|_| |_| |_|_.__/ \____|____/|_|  |_|_____|"""
        self.info(title)
        self.info(f"{__version__}\n")

    # pylint: disable-next=too-many-branches
    def log_model(self, model):
        r"""Log the relevant model properties

        Parameters
        ----------
        log : ``GDMLLogger``
            Log object to write to.
        model : :obj:`dict`
            Task or model.
        """
        d_type = model["type"]
        if d_type == "t":
            self.info("Task properties")
            self.info("---------------\n")
        elif d_type == "m":
            self.info("Model properties")
            self.info("----------------\n")
        else:
            raise ValueError(f"log_model does not support {d_type} type")
        if "z" in model:
            Z = model["z"]
        elif "Z" in model:
            Z = model["Z"]
        else:
            raise ValueError("z or Z does not exist in model")
        atom_string = "".join(atoms_by_element(Z))
        self.info("Atoms : %d", len(Z))
        self.info("Elements : %s", atom_string)
        try:
            self.info("n_train : %d", len(model["idxs_train"]))
            self.info("n_valid : %d", len(model["idxs_valid"]))
        except KeyError:
            self.info("n_train : %d", model["n_train"])
            self.info("n_valid : %d", model["n_valid"])
        self.info("sigma : %r", model["sig"])
        self.info("lambda : %r", model["lam"])
        if "perms" in model.keys():
            n_sym = model["perms"].shape[0]
            self.info("Symmetries : %d", n_sym)
        else:
            self.info("use_sym : %r", model["use_sym"])
        self.info("use_E : %r", model["use_E"])
        if d_type == "t":
            self.info("use_E_cstr : %r", model["use_E_cstr"])
        elif d_type == "m":
            if "alphas_E" in model.keys():  # pylint: disable=simplifiable-if-statement
                use_E_cstr = True
            else:
                use_E_cstr = False
            self.info(f"use_E_cstr : {use_E_cstr}")
        self.info(f"use_cprsn : {model['use_cprsn']}")

    def t_start(self):
        r"""Record the time and return a random hash."""
        t_hash = random.getrandbits(128)
        self.t_hashes[t_hash] = time.time()
        return t_hash

    def t_stop(self, t_hash, message="Took {time} s", precision=5, level=20):
        r"""Determine timing from a hash.

        Parameters
        ----------
        t_hash : :obj:`str`
            Timing hash generated from ``TimeTracker.start()``.
        message : :obj:`str`, default: ``'Took {time} s'``
            Timing message to be written to log. ``'{time}'`` will be replaced
            with for the elapsed time.
        precision : :obj:`int`, default: ``5``
            Number of decimal points to print.
        level : :obj:`int`, default: ``20``
            Log level.
        """
        t_stop = time.time()
        t_elapsed = t_stop - self.t_hashes[t_hash]
        del self.t_hashes[t_hash]
        # pylint: disable-next=no-member
        self.log(level, message.replace("{time}", f"%.{precision}f" % t_elapsed))
        return t_elapsed
