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

"""Alchemical utilities."""

import logging
log = logging.getLogger(__name__)

class mbeAlchemyScale(object):
    """Scale many-body interactions according to an alchemical parameter.
    """

    def __init__(self, entity_id, order, factor):
        """
        Parameters
        ----------
        entity_id : :obj:`int`
            Entity ID to scale interactions of.
        order : :obj:`int`
            Many-body order interactions to scale.
        factor : :obj:`float`
            Scaling factor that should be between ``0.`` and ``1.``.
        """
        self.entity_id = entity_id
        self.order = order
        self.factor = factor
    
    def scale(self, data):
        """Scale energies and forces.

        Parameters
        ----------
        data : :obj:`float` or :obj:`numpy.ndarray`
        """
        return self.factor*data
