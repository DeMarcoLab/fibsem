"""
Contains class Points for basic manipulation of points.

# Author: Vladan Lucic (Max Planck Institute of Biochemistry)
# $Id: points.py 1292 2016-04-27 10:35:30Z vladan $
"""

__version__ = "$Revision: 1292 $"


import numpy
import scipy
import scipy.linalg as linalg

class Points(object):
    """

    """

    def __init__(self, x=None):
        """
        """
        self.x = x

    def distance(self, index=None):
        """
        Finds Eucledian distance between N-dimensional points.

        Note: perhaps should go to another class.
        """

        # change to another reference 
        x_0 = self._shift(index)

        # calculate
        return numpy.sqrt(numpy.add.reduce(x_0**2, axis=-1))

    def angle(self, index=None, units='deg'):
        """
        Calculates angles between vectors formed by differences between points.

        Not finished
        """

        raise NotImplementedError("Sorry, this is still work in progress.")

        # change to another reference 
        x_0 = self._shift(index)

        # normalize
        length = self.distance(index)
        x_norm = x_0 / length[:,numpy.newaxis]

        # calculate
        res = numpy.arccos(numpy.inner(x_norm, x_norm)) 
        if units == 'deg':
            res = res * 180 / numpy.pi
        elif unitx == 'rad':
            pass
        else:
            raise ValueError("Sorry, did not understand units: " + units + ".")

        return res
        
    def _shift(self, index=None):
        """
        """

        if index is None:
            x_0 = self.x
        
        elif isinstance(index, int):
            x_0 = self.x - self.x[index]

        elif index == 'cm':
            x_0 = self.x - self.x.mean(axis=-2)

        else:
            raise ValueError("Sorry, did not understand index: "
                             + str(index) + ".")

        return x_0

