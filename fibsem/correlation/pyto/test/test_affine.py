"""

Tests module affine

# Author: Vladan Lucic
# $Id: test_affine.py 976 2013-06-04 20:08:03Z vladan $
"""

__version__ = "$Revision: 976 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.affine import Affine
from pyto.affine_2d import Affine2D

class TestAffine(np_test.TestCase):
    """
    """

    def setUp(self):

        # parallelogram, rotation, scale, exact
        self.d1 = [-1, 2]
        self.x1 = numpy.array([[0., 0], [2, 0], [2, 1], [0, 1]])
        self.y1 = numpy.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d1
        self.y1m = numpy.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d1

        # parallelogram, rotation, scale, not exact
        self.d2 = [-1, 2]
        self.x2 = numpy.array([[0.1, -0.2], [2.2, 0.1], [1.9, 0.8], [0.2, 1.1]])
        self.y2 = numpy.array([[0., 0], [4, 2], [3, 4], [-1, 2]]) + self.d2
        self.y2m = numpy.array([[0., 0], [-4, 2], [-3, 4], [1, 2]]) + self.d2

    def testIdentity(self):
        """
        Tests identity()
        """

        ndim = 3
        ident = Affine.identity(ndim=ndim)
        ident.decompose(order='qpsm')
        np_test.assert_almost_equal(ident.scale, numpy.ones(ndim))
        np_test.assert_almost_equal(ident.parity, 1)
        np_test.assert_almost_equal(ident.translation, numpy.zeros(shape=ndim))
        np_test.assert_almost_equal(ident.gl, numpy.identity(ndim))

    def testScale(self):
        """
        Tests getScale and setScale
        """

        af1m_desired = Affine.find(x=self.x1, y=self.y1m)
        af1m_changed = Affine.find(x=self.x1, y=self.y1m)
        af1m_changed.scale = [1, 2]
        np_test.assert_almost_equal(af1m_changed.s, [[1, 0], [0, 2]])
        np_test.assert_almost_equal(af1m_changed.scale, [1,2])
        np_test.assert_almost_equal(af1m_changed.q, af1m_desired.q)
        np_test.assert_almost_equal(af1m_changed.p, af1m_desired.p)
        np_test.assert_almost_equal(af1m_changed.m, af1m_desired.m)
        np_test.assert_almost_equal(af1m_changed.d, af1m_desired.d)

    def testFind(self):
        """
        Tests find() method
        """

        af = Affine.find(x=self.x1, y=self.y1m)
        desired = numpy.inner(self.x1, af.gl) + af.d
        np_test.assert_almost_equal(self.y1m, desired)
        
    def testFindTranslation(self):
        """
        Tests findTranslation()
        """

        af = Affine.findTranslation(x=numpy.array([[1,2,3], [2,3,4]]),
                                    y=numpy.array([[2,4,6], [3,6,9]]))
        
        np_test.assert_almost_equal(af.translation, [1., 2.5, 4.])
        af.decompose(order='qpsm')
        np_test.assert_almost_equal(af.scale, numpy.ones(3))
        np_test.assert_almost_equal(af.parity, 1)
        np_test.assert_almost_equal(af.gl, numpy.identity(3))

    def testFindTwoStep(self):
        """
        Tests findTwoStep()
        """

        # parallelogram, rotation, scale, exact
        af = Affine.findTwoStep(x=self.x1[0:1], y=self.y1[0:1],
                                x_gl=self.x1, y_gl=self.y1+3)
        af_desired = Affine.find(x=self.x1, y=self.y1)
        np_test.assert_almost_equal(af.gl, af_desired.gl)
        np_test.assert_almost_equal(af.d, af_desired.d)
        np_test.assert_almost_equal(af.glError, numpy.zeros_like(self.x1))
        np_test.assert_almost_equal(af.dError, numpy.zeros_like(self.x1[0:1]))
        np_test.assert_almost_equal(af.rmsErrorEst, 0)

        # parallelogram, rotation, scale, parity, exact
        af = Affine.findTwoStep(x=self.x1[0:2], y=self.y1m[0:2],
                                x_gl=self.x1, y_gl=self.y1m+[2,-3])
        af_desired = Affine.find(x=self.x1, y=self.y1m)
        np_test.assert_almost_equal(af.gl, af_desired.gl)
        np_test.assert_almost_equal(af.d, af_desired.d)
        np_test.assert_almost_equal(af.glError, numpy.zeros_like(self.x1))
        np_test.assert_almost_equal(af.dError, numpy.zeros_like(self.x1[0:2]))
        np_test.assert_almost_equal(af.rmsErrorEst, 0)

        # parallelogram, rotation, scale, parity, not exact
        af = Affine.findTwoStep(x=self.x2, y=self.y2m,
                                x_gl=self.x2, y_gl=self.y2m+[2,-3])
        af_desired = Affine.find(x=self.x2, y=self.y2m)
        np_test.assert_almost_equal(af.gl, af_desired.gl)
        np_test.assert_almost_equal(af.d, af_desired.d)
        np_test.assert_almost_equal(af.rmsErrorEst, af_desired.rmsError, 
                                    decimal=0)

    def testDecompose(self):
        """
        Tests decompose (decomposeQR and decomposeSV) and composeGl
        """

        repeat = 10
        for i in range(repeat):
 
            # initialize 3x3 random array
            gl = numpy.random.random((3,3))

            # check qpsm 
            af = Affine(gl=gl)
            af.decompose(order='qpsm')
            self.checkQRDecompose(af)
            af.gl = None
            new_gl = af.composeGl(order='qpsm')
            np_test.assert_almost_equal(new_gl, gl)

            # check psmq 
            af = Affine(gl=gl)
            af_1 = Affine()
            q, p, s, m = af_1.decompose(order='psmq', gl=gl)
            af_1.q = q
            af_1.p = p
            af_1.s = s
            af_1.m = m
            self.checkQRDecompose(af_1)
            af_2 = Affine()
            gl_2 = af_2.composeGl(order='psmq', q=q, p=p, s=s, m=m)
            np_test.assert_almost_equal(gl_2, gl)

            # check usv 
            af = Affine(gl=gl)
            af.decompose(order='usv')
            self.checkSVDecompose(af)
            af_1 = Affine()
            af_1.u = af.u
            af_1.s = af.s
            af_1.p = af.p
            af_1.v = af.v
            new_gl = af_1.composeGl(order='usv')
            np_test.assert_almost_equal(new_gl, gl)

            # initialize 4x4 random array
            gl = numpy.random.random((4,4))

            # check qpsm 
            af = Affine(gl=gl)
            af_1 = Affine()
            q, p, s, m = af_1.decompose(order='qpsm', gl=gl)
            af_1.q = q
            af_1.p = p
            af_1.s = s
            af_1.m = m
            self.checkQRDecompose(af_1)
            af_2 = Affine()
            gl_2 = af_2.composeGl(order='qpsm', q=q, p=p, s=s, m=m)
            np_test.assert_almost_equal(gl_2, gl)

            # check psmq 
            af = Affine(gl=gl)
            af.decompose(order='psmq')
            self.checkQRDecompose(af)
            af.gl = None
            new_gl = af.composeGl(order='psmq')
            np_test.assert_almost_equal(new_gl, gl)

            # check psmq 
            af = Affine(gl=gl)
            af_1 = Affine()
            af_1.u, af_1.p, af_1.s, af_1.v = af_1.decompose(order='usv', gl=gl)
            self.checkSVDecompose(af_1)
            af_2 = Affine()
            af_2.u = af_1.u
            af_2.s = af_1.s
            af_2.p = af_1.p
            af_2.v = af_1.v
            new_gl = af_2.composeGl(order='usv')
            np_test.assert_almost_equal(new_gl, gl)

    def checkQRDecompose(self, af):
        """
        Check properties of QR decomposition
        """

        size = af.q.shape[0]

        # q
        np_test.assert_almost_equal(scipy.linalg.det(af.q), 1)
        ortho_0, ortho_1 = self.checkOrtho(af.q)
        np_test.assert_almost_equal(ortho_0, numpy.identity(size))
        np_test.assert_almost_equal(ortho_1, numpy.identity(size))

        # p
        np_test.assert_equal(numpy.abs(af.p), numpy.identity(size))
        p_diag = af.p.diagonal()
        if p_diag[af.parity_axis] == 1:
            np_test.assert_equal((p_diag==1).all(), True)
        else:
            np_test.assert_equal(numpy.count_nonzero(~(p_diag==1)), 1)

        # s
        np_test.assert_equal((af.s > 0)*1., numpy.identity(size))
        np_test.assert_equal((af.s.diagonal() >= 0).all(), True)

        # m
        np_test.assert_almost_equal(af.m.diagonal(), numpy.ones(size))
        for i in range(size):
            for j in range(i):
                np_test.assert_almost_equal(af.m[i,j], 0)

    def checkSVDecompose(self, af):
        """
        Check properties of singular value decomposition
        """

        size = af.u.shape[0]

        # u
        np_test.assert_almost_equal(scipy.linalg.det(af.u), 1)
        ortho_0, ortho_1 = self.checkOrtho(af.u)
        np_test.assert_almost_equal(ortho_0, numpy.identity(size))
        np_test.assert_almost_equal(ortho_1, numpy.identity(size))

        # v
        np_test.assert_almost_equal(scipy.linalg.det(af.v), 1)
        ortho_0, ortho_1 = self.checkOrtho(af.v)
        np_test.assert_almost_equal(ortho_0, numpy.identity(size))
        np_test.assert_almost_equal(ortho_1, numpy.identity(size))

        # p
        np_test.assert_equal(numpy.abs(af.p), numpy.identity(size))
        p_diag = af.p.diagonal()
        if p_diag[af.parity_axis] == 1:
            np_test.assert_equal((p_diag==1).all(), True)
        else:
            np_test.assert_equal(numpy.count_nonzero(~(p_diag==1)), 1)
            
        # s
        np_test.assert_equal((af.s > 0)*1., numpy.identity(size))
        np_test.assert_equal((af.s.diagonal() >= 0).all(), True)

    def testInverse(self):
        """
        Tests inverse method
        """

        #################################################
        #
        # parallelogram, scale, rotation, parity, exact
        #

        # 
        af = Affine.find(x=self.x1, y=self.y1m)

        # test inverse
        af_inverse = af.inverse()
        np_test.assert_almost_equal(numpy.dot(af.gl, af_inverse.gl),
                                    numpy.identity(2))
        afi = Affine.find(x=self.y1m, y=self.x1)
        np_test.assert_almost_equal(af_inverse.gl, afi.gl)
        np_test.assert_almost_equal(af_inverse.d, afi.d)
        np_test.assert_almost_equal(self.x1, af_inverse.transform(self.y1m))

        # error
        np_test.assert_almost_equal(af_inverse.error, afi.error)
        np_test.assert_almost_equal(af_inverse.rmsError, afi.rmsError)

        #################################################
        #
        # parallelogram, scale, rotation, parity, not exact
        #
        # Note: only approximate comparisons because inverse of an optimal
        # (least squares) x->y transformation is not the optimal y->x.
        
        af = Affine.find(x=self.x2, y=self.y2m)

        # test inverse
        af_inverse = af.inverse()
        np_test.assert_almost_equal(numpy.dot(af.gl, af_inverse.gl),
                                    numpy.identity(2))
        afi = Affine.find(x=self.y2m, y=self.x2)
        np_test.assert_almost_equal(af_inverse.gl, afi.gl, decimal=1)
        np_test.assert_almost_equal(af_inverse.d, afi.d, decimal=1)
        np_test.assert_almost_equal(self.x2, af_inverse.transform(self.y2m),
                                    decimal=0)

        # error
        np_test.assert_almost_equal(af_inverse.error, afi.error, decimal=1)
        np_test.assert_almost_equal(af_inverse.rmsError, afi.rmsError, 
                                    decimal=1)        

    def testTransform(self):
        """
        Tests transform() method
        """
        af = Affine.find(x=self.x1, y=self.y1m)
        desired = numpy.inner(self.x1, af.gl) + af.d
        np_test.assert_almost_equal(af.transform(self.x1), desired)

    def testRemoveMasked(self):
        """
        Tests removeMasked()
        """

        x = numpy.array([[1,2], [3,4], [5,6]])
        x_mask = numpy.array([1, 0, 0])
        y = numpy.array([[2,4], [6,8], [10,12]])
        y_mask = numpy.array([0, 0, 1])
  
        data, total_mask = Affine.removeMasked(arrays=[x, y], 
                                               masks=(x_mask, y_mask))
        np_test.assert_equal(data[0], numpy.array([[3,4]]))
        np_test.assert_equal(data[1], numpy.array([[6,8]]))
        np_test.assert_equal(total_mask, numpy.array([1,0,1]))

        data, total_mask = Affine.removeMasked(arrays=[x, y])
        np_test.assert_equal(data[0], x)
        np_test.assert_equal(data[1], y)
        np_test.assert_equal(total_mask, numpy.array([0,0,0]))

    def checkOrtho(self, ar):
        """
        Calculates dot products between all rows and between all columns of a 
        matrix (arg ar). Used to check orthonormality of a matrix.

        Returns: the dot products if the form of two matrices
        """

        res_0 = numpy.zeros_like(ar) - 1.
        for i in range(ar.shape[0]):
            for j in range(ar.shape[0]):
                res_0[i, j] = numpy.dot(ar[i,:], ar[j,:])

        res_1 = numpy.zeros_like(ar) - 1.
        for i in range(ar.shape[0]):
            for j in range(ar.shape[0]):
                res_1[i, j] = numpy.dot(ar[:,i], ar[:,j])

        return res_0, res_1


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAffine)
    unittest.TextTestRunner(verbosity=2).run(suite)
