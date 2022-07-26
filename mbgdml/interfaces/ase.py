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

from ase.calculators.calculator import Calculator

class mbeCalculator(Calculator):
    """ASE calculator using the many-body expansion predictor in mbGDML.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self, mbe_pred, parameters=None, e_conv=1.0, f_conv=1.0, *args, **kwargs
    ):
        """
        Parameters
        ----------
        mbe_pred : :obj:`mbgdml.mbe.mbePredict`
            Initialized many-body expansion predictor.
        parameters : :obj:`dict`, optional
            System and calculation properties.
        e_conv : :obj:`float`, default: ``1.0``
            Model energy conversion factor to eV (required by ASE).
        f_conv : :obj:`float`, default: ``1.0``
            Model forces conversion factor to eV/A (required by ASE).
        """
        self.atoms = None
        self.name = 'mbGDML'
        self.results = {}

        self.mbe_pred = mbe_pred

        self.e_conv = e_conv
        self.f_conv = f_conv
        
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def calculate(self, atoms=None, *args, **kwargs):
        """Predicts energy and forces using many-body GDML models.
        """
        parameters = self.parameters
        entity_ids = parameters['entity_ids']
        comp_ids = parameters['comp_ids']

        e, f = self.mbe_pred.predict(
            atoms.get_atomic_numbers(), atoms.get_positions(),
            entity_ids, comp_ids
        )
        e = e[0]

        # Unit conversions
        e *= self.e_conv
        f *= self.f_conv

        self.results = {'energy': e, 'forces': f.reshape(-1, 3)}
