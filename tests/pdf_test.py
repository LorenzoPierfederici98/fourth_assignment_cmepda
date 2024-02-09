"""
Copyright (C) 2024  Lorenzo Pierfederici 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
# if the test is running in interactive mode (-i)
if sys.flags.interactive:
    plt.ion()

# This adds src to the list of directories the interpreter will search
# for the required module. main.py mustn't be moved from src
# directory
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from src.main import ProbabilityDensityFunction

class Testpdf(unittest.TestCase):
    """Class for unit-testing.
    """
    def _test_triangular_base(self, xmin=0., xmax=1.):
        """Unit test with a triangular distribution.
        """
        x = np.linspace(xmin, xmax, 101)
        y = 2. / (xmax - xmin)**2. * (x - xmin)
        pdf = ProbabilityDensityFunction(x, y)

        # Verify that the pdf normalization is one.
        norm = pdf.integral(xmin, xmax)
        self.assertAlmostEqual(norm, 1.0)

        # Verify that the pdf, evaluated on the input x-grid, matches the
        # input y values.
        delta = abs(pdf(x) - y)
        self.assertTrue((delta < 1e-10).all())

        plt.figure('pdf triangular')
        plt.plot(x, pdf(x))
        plt.xlabel('x')
        plt.ylabel('pdf(x)')

        plt.figure('cdf triangular')
        plt.plot(x, pdf.cdf(x))
        plt.xlabel('x')
        plt.ylabel('cdf(x)')

        plt.figure('ppf triangular')
        q = np.linspace(0., 1., 250)
        plt.plot(q, pdf.ppf(q))
        plt.xlabel('q')
        plt.ylabel('ppf(q)')

        plt.figure('Sampling triangular')
        rnd = pdf.rnd(1000000)
        plt.hist(rnd, bins=200)

    def test_triangular(self):
        """Uses _test_triangular_base to test triangular pdfs.
        """
        self._test_triangular_base(0., 1.)
        self._test_triangular_base(0., 2.)
        self._test_triangular_base(1., 2.)
        


if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)
