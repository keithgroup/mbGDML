# MIT License
# 
# Copyright (c) 2020, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

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

"""Analyses for mbGDML models."""

from mbgdml.data import structure
from ase import Atoms
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize


class similarity:

    def __init__(self):
        pass

    
    def prepare_structures(self, z, R):

        structures = []

        for struct_num in R:
            print('yes')


    def initialize_soap(self, atoms, rcut, nmax, lmax, sigma, **kwargs):

        atoms = list(set(atoms))  # Unique list of atomic species

        self.desc = SOAP(
            species=atoms,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            periodic=False,
            crossover=True,
            sparse=False
        )



        



def test():
    string_file = '/home/alex/Dropbox/keith/projects/mbgdml/data/analysis/similarity/test/h2o-lit-structures.xyz'

    test_structures = structure()
    test_structures.load_file(string_file)
    test = similarity()
    test.initialize_soap(
        test_structures.z,
        5.0,
        8,
        6,
        0.5
    )

    test.prepare_structures(test_structures.z, test_structures.R)

    return 'test'

run = test()