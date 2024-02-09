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
from scipy.interpolate import InterpolatedUnivariateSpline

# This adds src to the list of directories the interpreter will search
# for the required module. main.py mustn't be moved from src
# directory
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from src.main import ProbabilityDensityFunction

class Testpdf(unittest.TestCase):
    """Class for unit-testing.
    """
    def test_probability(self):
        """Tests that the probability calculated with the class ProbabilityDensityFunction is equal to the one calculated with splines.
        """
        x = np.linspace(0, 1, 101)
        y = 2 * x
        x1 = 0.3
        x2 = 0.5
        testing_pdf = InterpolatedUnivariateSpline(x, y)
        testing_integral = testing_pdf.integral(x1, x2)
        pdf = ProbabilityDensityFunction(x, y)
        integral = pdf.integral(x1, x2)
        self.assertAlmostEqual(integral, testing_integral)

if __name__ == '__main__':
    unittest.main()
