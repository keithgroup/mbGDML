# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
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

log = logging.getLogger(__name__)

class model(object):
    """A parent class for machine learning model objects.
    
    Attributes
    ----------
    md5 : :obj:`str`
        A property that creates a unique MD5 hash for the model. This is
        primarily only used in the creation of predict sets.
    nbody_order : :obj:`int`
        What order of :math:`n`-body contributions does this model predict?
        This is easily determined by taking the length of component IDs for the
        model.
    """

    def __init__(self, criteria_desc_func=None, criteria_cutoff=None):
        """
        Parameters
        ----------
        criteria_desc : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc`` where the mlModel will not predict the
            :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        self.criteria_desc_func = criteria_desc_func
        # Make sure cutoff is a single value (weird extraction from npz).
        if isinstance(criteria_cutoff, np.ndarray):
            if len(criteria_cutoff) == 0:
                criteria_cutoff = None
            elif len(criteria_cutoff) == 1:
                criteria_cutoff = criteria_cutoff[0]
        self.criteria_cutoff = criteria_cutoff
