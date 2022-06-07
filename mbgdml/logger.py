# MIT License
# 
# Copyright (c) 2020-2022, Alex M. Maldonado
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
import numpy as np
import random
import time
from . import __version__
from .utils import z_to_element

class timeTracker:
    """Simple way to keep track of multiple timings."""
    def __init__(self):
        self.t_hashes = {}

    def t_start(self):
        """Record the time and return a random hash."""
        t_hash = random.getrandbits(128)
        self.t_hashes[t_hash] = time.time()
        return t_hash
    
    def t_stop(
        self, t_hash, message='Took {time} s', precision=5, level=20
    ):
        """Determine timing from a hash.
        
        Parameters
        ----------
        t_hash : :obj:`str`
            Timing hash generated from ``timeTracker.start()``.
        log : ``GDMLLogger``
            Log object to write to.
        message : :obj:`str`, default: ``'Took {time} s'``
            Timing message to be written to log. ``'{time}'`` will be replaced
            with for the elapsed time.
        precision : :obj:`int`, default: ``5``
            Number of decimal points to print time.
        """
        t_stop = time.time()
        t_elapsed = t_stop - self.t_hashes[t_hash]
        del self.t_hashes[t_hash]
        self.log(
            level, message.replace('{time}', f'%.{precision}f' % t_elapsed)
        )
        return t_elapsed

class GDMLLogger(logging.Logger, timeTracker):

    level = logging.INFO
    
    def __init__(self, name):

        logging.Logger.__init__(self, name, self.level)
        timeTracker.__init__(self)

        # only display levelname and message
        formatter = logging.Formatter('%(message)s')

        # this handler will write to sys.stderr by default
        hd = logging.StreamHandler()
        hd.setFormatter(formatter)
        hd.setLevel(self.level)

        self.addHandler(hd)

        return

    def log_array(self, array, level=20):
        if isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        arr_str = np.array2string(
            array, separator=', '
        )
        self.log(level, arr_str)

    def log_package(self):
        title = r"""           _      ____ ____  __  __ _     
 _ __ ___ | |__  / ___|  _ \|  \/  | |    
| '_ ` _ \| '_ \| |  _| | | | |\/| | |    
| | | | | | |_) | |_| | |_| | |  | | |___ 
|_| |_| |_|_.__/ \____|____/|_|  |_|_____|"""
        self.info(title)
        self.info(f'{__version__}\n')

    def log_model(self, model):
        """Log the relevant model properties
        
        Parameters
        ----------
        log : ``GDMLLogger``
            Log object to write to.
        model : :obj:`dict`
            Task or model.
        """
        d_type = model['type']
        if d_type == 't':
            self.info('Task properties')
            self.info('---------------\n')
        elif d_type == 'm':
            self.info('Model properties')
            self.info('----------------\n')
        else:
            raise ValueError(f'log_model does not support {d_type} type')
        z = model["z"]
        atom_string = ' '.join(
            z_to_element[i] for i in z
        )
        self.info(f'Atoms : {len(z)}')
        self.info(f'Elements : {atom_string}')
        try:
            self.info(f'n_train : {len(model["idxs_train"])}')
            self.info(f'n_valid : {len(model["idxs_valid"])}')
        except KeyError:
            self.info(f'n_train : {model["n_train"]}')
            self.info(f'n_valid : {model["n_valid"]}')
        self.info(f'sigma : {model["sig"]}')
        self.info(f'lambda : {model["lam"]}')
        if 'perms' in model.keys():
            n_sym = model['perms'].shape[0]
            self.info(f'Symmetries : {n_sym}')
        else:
            self.info(f'use_sym : {model["use_sym"]}')
        self.info(f'use_E : {model["use_E"]}')
        if d_type == 't':
            self.info(f'use_E_cstr : {model["use_E_cstr"]}')
        elif d_type == 'm':
            if 'alphas_E' in model.keys():
                use_E_cstr = True
            else:
                use_E_cstr = False
            self.info(f'use_E_cstr : {use_E_cstr}')
        self.info(f'use_cprsn : {model["use_cprsn"]}')
