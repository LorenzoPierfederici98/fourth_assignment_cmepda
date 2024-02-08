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

import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """Class describing pdfs.
    """

    def __init__(self, x, y):
        InterpolatedUnivariateSpline.__init__(self, x, y)
        # Comulative density function cdf
        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf)
        # ppf is the inverse of cdf: x = ppf(q), q uniformly distributed in [0,1], x distributed
        # according to the pdf
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]
        self.ppf = InterpolatedUnivariateSpline(xppf, yppf)

    def probability(self, x1, x2):
        """Returns the probability for a random variabile to be included between x1 and x2.
        """
        if x1>x2:
            return self.cdf(x1)-self.cdf(x2)
        else:
            return self.cdf(x2)-self.cdf(x1)

    def rnd(self, size=1000):
        """Returns an array of size size of random values form the pdf.
        """
        return self.ppf(np.random.uniform(size=size))

if __name__ == '__main__':
    x = np.linspace(0, 1, 101)
    y = 2 * x
    pdf = ProbabilityDensityFunction(x, y)
