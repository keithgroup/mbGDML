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

import numpy as np

import logging
log = logging.getLogger(__name__)

def mae(errors):
    """Mean absolute error.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`

    Return
    ------
    :obj:`float`
        Mean absolute error.
    """
    return np.mean(np.abs(errors.flatten()))

def mse(errors):
    """Mean square error.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`

    Return
    ------
    :obj:`float`
        Mean square error.
    """
    return np.mean(np.abs(errors.flatten()))

def rmse(errors):
    """Mean square error.

    Parameters
    ----------
    errors : :obj:`numpy.ndarray`

    Return
    ------
    :obj:`float`
        Mean square error.
    """
    return np.sqrt(mse(errors))

def loss_f_rmse(results):
    """Returns the force RMSE.

    Parameters
    ----------
    results : :obj:`dict`
        Validation results which contain force and energy MAEs and RMSEs.
    """
    return rmse(results['force'])

def loss_f_e_weighted_mse(results, rho, n_atoms):
    r"""Computes a combined energy and force loss function.

    .. math::

        l = \frac{\rho}{Q} \left\Vert E - \hat{E} \right\Vert^2
        + \frac{1}{n_{atoms} Q} \sum_{i=0}^{n_{atoms}}
        \left\Vert \bf{F}_i - \widehat{\bf{F}}_i \right\Vert^2,
    
    where :math:`\rho` is a trade-off between energy and force errors,
    :math:`Q` is the number of validation structures, :math:`\Vert \ldots \Vert`
    is the norm, and :math:`\widehat{\;}` is the model prediction of the
    property.

    Parameters
    ----------
    results : :obj:`dict`
        Validation results which contain force and energy MAEs and RMSEs.
    rho : :obj:`float`
        Energy and force trade-off, :math:`\rho`. A recommended value would be
        in the range of ``0.01`` to ``0.1``. Larger values place a higher
        importance on energy accuracy.
    n_atoms : :obj:`int`
        Number of atoms.
    
    Returns
    -------
    :obj:`float`
        Validation loss.
    """
    F_mse = mse(results['force'])
    E_mse = mse(results['energy'])
    return rho*E_mse + (1/n_atoms)*F_mse
