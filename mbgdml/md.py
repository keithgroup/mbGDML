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
from cclib.parser.utils import convertor
from ase.calculators.calculator import Calculator as ASECalculator
from .predict import mbPredict


class mbGDML_ASE_Calculator(ASECalculator):
    """Initializes mbGDML-ASE calculator with models and units.
    
    Parameters
    -----------
    model_paths : :obj:`list`
        Paths of all models to be used for GDML prediction.
    e_unit_model : :obj:`str`, optional
        Specifies the units of energy prediction for the GDML models. Defaults
        to 'kcal/mol'.
    r_unit_model : :obj:`str`, optional
        Specifies the distance units for GDML models. Defaults to 'Angstrom'.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self, elements, model_paths, entity_ids, comp_ids,
        e_unit_model='kcal/mol', r_unit_model='Angstrom', *args, **kwargs
    ):
        """

        Parameters
        ----------
        """

        # TODO logging?
        self.atoms = None
        self.elements = elements
        self.entity_ids = entity_ids
        self.comp_ids = comp_ids

        #self.load_models(model_paths)
        self.gdml_predict = mbPredict(model_paths)

        self.e_unit_model = e_unit_model
        self.r_unit_model = r_unit_model
        
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

    def calculate(self, atoms=None, *args, **kwargs):
        """Predicts energy and forces using many-body GDML models.
        """

        super(mbGDMLCalculator, self).calculate(
            atoms, *args, **kwargs
        )

        r = np.array(atoms.get_positions())
        e, f = self.gdml_predict.predict(
            self.elements, r, self.entity_ids, self.comp_ids
        )
        e = e[0]

        # convert model units to ASE default units (eV and Ang)
        if self.e_unit_model != 'eV':
            e *= convertor(1, self.e_unit_model, 'eV')
            f *= convertor(1, self.e_unit_model, 'eV')

        if self.r_unit_model != 'Angstrom':
            f /= convertor(1, self.r_unit_model, 'Angstrom')

        self.results = {'energy': e, 'forces': f.reshape(-1, 3)}
