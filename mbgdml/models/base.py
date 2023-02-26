# MIT License
#
# Copyright (c) 2022-2023, Alex M. Maldonado
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

from ..logger import GDMLLogger

log = GDMLLogger(__name__)


class Model:
    r"""A parent class for machine learning model objects.

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

    def __init__(self, criteria=None):
        """
        Parameters
        ----------
        criteria : :obj:`mbgdml.descriptors.Criteria`, default: :obj:`None`
            Initialized descriptor criteria for accepting a structure based on
            a descriptor and cutoff.
        """
        self.criteria = criteria
