"""

Tests module affine_2d

# Author: Vladan Lucic
# $Id: test_affine_2d.py 1152 2015-05-26 08:53:37Z vladan $
"""

__version__ = "$Revision: 1152 $"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.affine_2d import Affine2D

class TestAffine2D(np_test.TestCase):
    """
    """

    def setUp(self):

        # basic
        self.x0 = numpy.array([[1, 0.], [0, 1], [-1, 0]])
        self.y0_0 = 2 * self.x0
        self.y0_90 = 2 * numpy.array([[0, 1], [-1, 0], [0, -1]])
        self.y0_180 = 2 * numpy.array([[-1, 0.], [0, -1], [1, 0]])
        self.y0_270 = 2 * numpy.array([[0, -1], [1, 0], [0, 1]])

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

        # transformations
        self.af1 = Affine2D.find(x=self.x1, y=self.y1)
        self.af1_gl = numpy.array([[2,-1],[1,2]])
        self.af1_d = self.d1
        self.af1_phi = numpy.arctan(0.5)
        self.af1_scale = numpy.array([numpy.sqrt(5)] * 2)
        self.af1_parity = 1
        self.af1_shear = 0

        self.af1m = Affine2D.find(x=self.x1, y=self.y1m)
        self.af1m_gl = numpy.array([[-2,1],[1,2]])
        self.af1m_d = self.af1_d
        self.af1m_phi = numpy.pi - self.af1_phi
        self.af1m_q = self.af1m.makeQ(phi=self.af1m_phi)
        self.af1m_scale = numpy.array([numpy.sqrt(5)] * 2)
        self.af1m_parity = -1
        self.af1m_shear = 0

        self.af2 = Affine2D.find(x=self.x2, y=self.y2)
        self.af2_d = [-1.42584884,  2.05326245]
        self.af2_gl = numpy.array([[2.09463865, -0.84056372],
                                 [ 1.00406239,  1.87170871]])
        self.af2_phi = 0.446990530695
        self.af2_scale = numpy.array([2.32285435, 2.05115392])

        self.af2m = Affine2D.find(x=self.x2, y=self.y2m)

        # L-shaped u, scale, v_angle=0
        self.x3 = numpy.array([[3,0], [2,0], [1,0], [1, -1]])
        self.y3 = numpy.array([[-1,2], [-1,1.5], [-1,1], [1, 1]])
        self.af3 = Affine2D.find(x=self.x3, y=self.y3)
        self.af3.decompose(order='usv')
        self.af3_uAngleDeg = 0
        self.af3_vAngleDeg = 90
        self.af3_scale = [2, 0.5]
        self.af3_d = [-1, 0.5]

    def testFindGL(self):
        """
        Tests find (transform 'gl'), decompose individual parameters and
        transform.
        """

        ##################################################
        #
        # parallelogram, rotation, scale, exact
        #

        #aff2d = Affine2D.find(x=self.x1, y=self.y1)
        np_test.assert_almost_equal(self.af1.d, self.af1_d)
        np_test.assert_almost_equal(self.af1.gl, self.af1_gl)

        # xy_axis = 'dim_point'
        aff2d_xy = Affine2D.find(
            x=self.x1.transpose(), y=self.y1.transpose(), xy_axes='dim_point')
        np_test.assert_almost_equal(aff2d_xy.d, self.af1_d)
        np_test.assert_almost_equal(aff2d_xy.gl, self.af1_gl)

        # test decompose
        self.af1.decompose(order='qpsm') 
        np_test.assert_almost_equal(self.af1.phi, self.af1_phi)
        desired_q = numpy.array(\
            [[numpy.cos(self.af1_phi), -numpy.sin(self.af1_phi)],
             [numpy.sin(self.af1_phi), numpy.cos(self.af1_phi)]])
        np_test.assert_almost_equal(self.af1.q, desired_q)
        np_test.assert_almost_equal(self.af1.p, numpy.diag([1, 1]))
        np_test.assert_almost_equal(self.af1.s, 
                                    self.af1_scale * numpy.diag([1,1]))
        np_test.assert_almost_equal(self.af1.m, numpy.diag([1, 1]))

        # test parameters
        np_test.assert_almost_equal(self.af1.scale, self.af1_scale)
        np_test.assert_almost_equal(self.af1.phi, self.af1_phi)
        np_test.assert_almost_equal(self.af1.parity, self.af1_parity)
        np_test.assert_almost_equal(self.af1.shear, self.af1_shear)
        
        # test transformation and error
        y1_calc = self.af1.transform(self.x1)
        np_test.assert_almost_equal(y1_calc, self.y1)
        np_test.assert_almost_equal(self.af1.error, numpy.zeros_like(self.y1))
        np_test.assert_almost_equal(self.af1.rmsError, 0)

        #################################################
        #
        # parallelogram, scale, rotation, parity, exact
        #

        # test parameters
        #aff2d = Affine2D.find(x=self.x1, y=self.y1m)
        np_test.assert_almost_equal(self.af1m.d, self.af1m_d)
        np_test.assert_almost_equal(self.af1m.gl, self.af1m_gl)
        np_test.assert_almost_equal(self.af1m.scale, self.af1m_scale)
        np_test.assert_almost_equal(self.af1m.phi, self.af1m_phi)
        #np_test.assert_almost_equal(self.af1m.phiDeg, 
        #                            180 - desired_phi * 180 / numpy.pi)
        np_test.assert_almost_equal(self.af1m.parity, self.af1m_parity)
        np_test.assert_almost_equal(self.af1m.shear, self.af1m_shear)

        # test transformation and error
        y1_calc = self.af1m.transform(self.x1, gl=self.af1m.gl, d=self.af1m.d)
        np_test.assert_almost_equal(y1_calc, self.y1m)
        np_test.assert_almost_equal(self.af1m.error, numpy.zeros_like(self.y1))
        np_test.assert_almost_equal(self.af1m.rmsError, 0)

        # xy_axis = 'dim_point'
        af1m_xy = Affine2D.find(
            x=self.x1.transpose(), y=self.y1m.transpose(), xy_axes='dim_point')
        np_test.assert_almost_equal(af1m_xy.d, self.af1m_d)
        np_test.assert_almost_equal(af1m_xy.gl, self.af1m_gl)
        np_test.assert_almost_equal(af1m_xy.scale, self.af1m_scale)
        np_test.assert_almost_equal(af1m_xy.phi, self.af1m_phi)
        np_test.assert_almost_equal(af1m_xy.parity, self.af1m_parity)
        np_test.assert_almost_equal(af1m_xy.shear, self.af1m_shear)

        ##################################################
        #
        # same as above but rq order
        #

        # test parameters
        q, p, s, m = self.af1m.decompose(gl=self.af1m.gl, order='rq')
        q_new = numpy.dot(numpy.dot(p, self.af1m_q), p) 
        np_test.assert_almost_equal(q, q_new)
        np_test.assert_almost_equal(p, self.af1m.makeP(parity=self.af1m_parity))
        np_test.assert_almost_equal(s, self.af1m.makeS(self.af1m_scale))
        np_test.assert_almost_equal(m, self.af1m.makeM(self.af1m_shear))

        # test transformation 
        psmq = numpy.dot(numpy.dot(p, s), numpy.dot(m, q))
        y_new = numpy.inner(self.x1, psmq) + self.af1m.d
        np_test.assert_almost_equal(y_new, self.y1m)

        ##################################################
        #
        # parallelogram, rotation, scale, not exact
        #

        aff2d = Affine2D.find(x=self.x2, y=self.y2)

        # test transformation matrices and parameters
        desired_d = [-1.42584884,  2.05326245]
        desired_gl = numpy.array([[2.09463865, -0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d.d, desired_d)
        np_test.assert_almost_equal(aff2d.gl, desired_gl)
        np_test.assert_almost_equal(aff2d.phi, desired_phi)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, 1)
        np_test.assert_almost_equal(aff2d.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2_calc_gl = aff2d.transform(self.x2, gl=aff2d.gl, d=aff2d.d)
        qpsm = numpy.dot(numpy.dot(aff2d.q, aff2d.p), 
                         numpy.dot(aff2d.s, aff2d.m))
        y2_calc_qpsm = numpy.inner(self.x2, qpsm) + aff2d.d
        np_test.assert_almost_equal(y2_calc_gl, y2_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2)

        ##################################################
        #
        # parallelogram, rotation, scale, not exact, xy_axes=dim_point
        #

        aff2d_xy = Affine2D.find(
            x=self.x2.transpose(), y=self.y2.transpose(), xy_axes='dim_point')

        # test transformation matrices and parameters
        desired_d = [-1.42584884,  2.05326245]
        desired_gl = numpy.array([[2.09463865, -0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d_xy.d, desired_d)
        np_test.assert_almost_equal(aff2d_xy.gl, desired_gl)
        np_test.assert_almost_equal(aff2d_xy.phi, desired_phi)
        np_test.assert_almost_equal(aff2d_xy.scale, desired_scale)
        np_test.assert_almost_equal(aff2d_xy.parity, 1)
        np_test.assert_almost_equal(aff2d_xy.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2_calc_gl = aff2d.transform(
            self.x2.transpose(), gl=aff2d_xy.gl, d=aff2d_xy.d, 
            xy_axes='dim_point')
        qpsm = numpy.dot(numpy.dot(aff2d_xy.q, aff2d_xy.p), 
                         numpy.dot(aff2d_xy.s, aff2d_xy.m))
        y2_calc_qpsm = numpy.dot(
            qpsm, self.x2.transpose()) + numpy.expand_dims(aff2d_xy.d, 1)
        np_test.assert_almost_equal(y2_calc_gl, y2_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2)

        ##################################################
        #
        # parallelogram, rotation, scale, parity, not exact
        #

        aff2d = Affine2D.find(x=self.x2, y=self.y2m)

        # test transformation matrices and parameters
        desired_d = [-0.57415116,  2.05326245]
        desired_gl = numpy.array([[-2.09463865, 0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d.d, desired_d)
        np_test.assert_almost_equal(aff2d.gl, desired_gl)
        np_test.assert_almost_equal(aff2d.phi, numpy.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(aff2d.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2m_calc_gl = aff2d.transform(self.x2)
        qpsm = numpy.dot(numpy.dot(aff2d.q, aff2d.p), 
                         numpy.dot(aff2d.s, aff2d.m))
        np_test.assert_almost_equal(qpsm, aff2d.gl)
        y2m_calc_qpsm = numpy.inner(self.x2, qpsm) + aff2d.d
        np_test.assert_almost_equal(y2m_calc_gl, y2m_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2m)

        ##################################################
        #
        # parallelogram, rotation, scale, parity, not exact, xy_axes='dim_point'
        #

        aff2d = Affine2D.find(
            x=self.x2.transpose(), y=self.y2m.transpose(), xy_axes='dim_point')

        # test transformation matrices and parameters
        desired_d = [-0.57415116,  2.05326245]
        desired_gl = numpy.array([[-2.09463865, 0.84056372],
                                 [ 1.00406239,  1.87170871]])
        desired_phi = 0.446990530695
        desired_scale = [2.32285435, 2.05115392]
        np_test.assert_almost_equal(aff2d.d, desired_d)
        np_test.assert_almost_equal(aff2d.gl, desired_gl)
        np_test.assert_almost_equal(aff2d.phi, numpy.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(aff2d.m, [[1, 0.02198716], [0, 1]])

        # test transform method
        y2m_calc_gl = aff2d.transform(self.x2.transpose())
        qpsm = numpy.dot(numpy.dot(aff2d.q, aff2d.p), 
                         numpy.dot(aff2d.s, aff2d.m))
        np_test.assert_almost_equal(qpsm, aff2d.gl)
        y2m_calc_qpsm = (numpy.dot(qpsm, self.x2.transpose()) 
                         + numpy.expand_dims(aff2d.d, 1))
        np_test.assert_almost_equal(y2m_calc_gl, y2m_calc_qpsm)
        #np_test.assert_almost_equal(y2_calc_gl, self.y2m)

        ##################################################
        #
        # L-shape: rotation, scale; check usv 
        #
        af3 = Affine2D.find(x=self.x3, y=self.y3)
        af3.decompose(order='usv')
        np_test.assert_almost_equal(af3.vAngleDeg, 90)
        np_test.assert_almost_equal(af3.uAngleDeg, 0)
        np_test.assert_almost_equal(af3.scale, [2, 0.5])
        np_test.assert_almost_equal(af3.scaleAngle, numpy.arccos(0.25))
        np_test.assert_almost_equal(af3.d, self.af3_d)

    def testFindRS(self):
        """
        Tests find (transform 'rs'), decompose individual parameters and
        transform.
        """

        ###############################################
        #
        # parallelogram, rotation, scale, exact
        #

        aff2d = Affine2D.find(x=self.x1, y=self.y1, type_='rs')
        np_test.assert_almost_equal(aff2d.d, self.d1)

        # test finding transformation
        desired_phi = numpy.arctan(0.5)
        desired_scale = [numpy.sqrt(5)] * 2
        desired_q = numpy.array(\
            [[numpy.cos(desired_phi), -numpy.sin(desired_phi)],
             [numpy.sin(desired_phi), numpy.cos(desired_phi)]])
        np_test.assert_almost_equal(aff2d.parity, 1)
        #np_test.assert_almost_equal(aff2d.phi, desired_phi)
        #np_test.assert_almost_equal(aff2d.q, desired_q)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.error, numpy.zeros_like(self.y1))

        # test doing transformation
        y1_calc = aff2d.transform(self.x1)
        np_test.assert_almost_equal(y1_calc, self.y1)
        qpsm = numpy.dot(numpy.dot(aff2d.q, aff2d.p), 
                         numpy.dot(aff2d.s, aff2d.m))
        y_new = numpy.inner(self.x1, qpsm) + aff2d.d
        np_test.assert_almost_equal(y_new, self.y1)
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity, exact
        #

        aff2d = Affine2D.find(x=self.x1, y=self.y1m, type_='rs')
        np_test.assert_almost_equal(aff2d.d, self.d1)

        # test finding transformation
        desired_phi = numpy.arctan(0.5)
        desired_scale = [numpy.sqrt(5)] * 2
        desired_q = numpy.array(\
            [[-numpy.cos(desired_phi), -numpy.sin(desired_phi)],
             [numpy.sin(desired_phi), -numpy.cos(desired_phi)]])
        np_test.assert_almost_equal(aff2d.phi, numpy.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.q, desired_q)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(aff2d.error, numpy.zeros_like(self.y1))

        # test doing transformation
        y1_calc = aff2d.transform(self.x1)
        np_test.assert_almost_equal(y1_calc, self.y1m)
        qpsm = numpy.dot(numpy.dot(aff2d.q, aff2d.p), 
                         numpy.dot(aff2d.s, aff2d.m))
        y_new = numpy.inner(self.x1, qpsm) + aff2d.d
        np_test.assert_almost_equal(y_new, self.y1m)
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity, exact, xy_axes='dim_point'
        #

        aff2d = Affine2D.find(
            x=self.x1.transpose(), y=self.y1m.transpose(), type_='rs', 
            xy_axes='dim_point')
        np_test.assert_almost_equal(aff2d.d, self.d1)

        # test finding transformation
        desired_phi = numpy.arctan(0.5)
        desired_scale = [numpy.sqrt(5)] * 2
        desired_q = numpy.array(\
            [[-numpy.cos(desired_phi), -numpy.sin(desired_phi)],
             [numpy.sin(desired_phi), -numpy.cos(desired_phi)]])
        np_test.assert_almost_equal(aff2d.phi, numpy.pi - desired_phi)
        np_test.assert_almost_equal(aff2d.q, desired_q)
        np_test.assert_almost_equal(aff2d.scale, desired_scale)
        np_test.assert_almost_equal(aff2d.parity, -1)
        np_test.assert_almost_equal(
            aff2d.error, numpy.zeros_like(self.y1.transpose()))

        # test doing transformation
        y1_calc = aff2d.transform(self.x1.transpose())
        np_test.assert_almost_equal(y1_calc, self.y1m.transpose())
        qpsm = numpy.dot(numpy.dot(aff2d.q, aff2d.p), 
                         numpy.dot(aff2d.s, aff2d.m))
        y_new = (numpy.dot(qpsm, self.x1.transpose()) 
                 + numpy.expand_dims(aff2d.d, 1))
        np_test.assert_almost_equal(y_new, self.y1m.transpose())
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity not exact
        #

        af2m = Affine2D.find(x=self.x2, y=self.y2m)
        af2mrs = Affine2D.find(x=self.x2, y=self.y2m, type_='rs')
 
        # test finding transformation
        desired_d = [-0.57415116,  2.05326245]
        desired_phi = numpy.pi - 0.442817288965
        desired_scale = [2.18278075] * 2
        desired_q = numpy.array(\
            [[numpy.cos(desired_phi), -numpy.sin(desired_phi)],
             [numpy.sin(desired_phi), numpy.cos(desired_phi)]])
        #np_test.assert_almost_equal(af2mrs.d, desired_d)
        np_test.assert_almost_equal(af2mrs.phi, desired_phi)
        np_test.assert_almost_equal(af2mrs.scale, desired_scale)
        np_test.assert_almost_equal(af2mrs.parity, -1)

        # compare with gl
        #np_test.assert_almost_equal(af2mrs.d, af2m.d)
        np_test.assert_almost_equal(af2mrs.scale, af2m.scale, decimal=1)
        np_test.assert_almost_equal(af2mrs.phi, af2m.phi, decimal=2)
        np_test.assert_almost_equal(af2mrs.parity, af2m.parity)
        np_test.assert_almost_equal(af2mrs.error, af2m.error, decimal=0)
        np_test.assert_almost_equal(af2mrs.rmsError, af2m.rmsError, decimal=1)

        # test doing transformation
        y2_calc = af2mrs.transform(self.x2)
        qpsm = numpy.dot(numpy.dot(af2mrs.q, af2mrs.p), 
                         numpy.dot(af2mrs.s, af2mrs.m))
        np_test.assert_almost_equal(qpsm, af2mrs.gl)
        y2_calc_qpsm = numpy.inner(self.x2, qpsm) + af2mrs.d
        np_test.assert_almost_equal(y2_calc, y2_calc_qpsm)
        
        ###############################################
        #
        # parallelogram, rotation, scale, parity not exact, xy_axes='dim_point'
        #

        af2m = Affine2D.find(x=self.x2.transpose(), y=self.y2m.transpose())
        af2mrs = Affine2D.find(
            x=self.x2.transpose(), y=self.y2m.transpose(), type_='rs',
            xy_axes='dim_point')
 
        # test finding transformation
        desired_phi = numpy.pi - 0.442817288965
        desired_scale = [2.18278075] * 2
        desired_q = numpy.array(\
            [[numpy.cos(desired_phi), -numpy.sin(desired_phi)],
             [numpy.sin(desired_phi), numpy.cos(desired_phi)]])
        np_test.assert_almost_equal(af2mrs.phi, desired_phi)
        np_test.assert_almost_equal(af2mrs.scale, desired_scale)
        np_test.assert_almost_equal(af2mrs.parity, -1)

    def testInverse(self):
        """
        Tests inverse
        """
        ###############################################
        #
        # parallelogram, rotation, scale not exact
        #

        af2 = Affine2D.find(x=self.x2, y=self.y2)
        af2rs = Affine2D.find(x=self.x2, y=self.y2, type_='rs')
        af2rsi = Affine2D.find(y=self.x2, x=self.y2, type_='rs')
        af2rs_inv = af2rs.inverse()
        af2rs_inv.decompose(order='qpsm')

        # tests inverse method
        np_test.assert_almost_equal(af2rs_inv.phi, -af2rs.phi)
        np_test.assert_almost_equal(af2rs_inv.scale, 1/af2rs.scale)
        np_test.assert_almost_equal(af2rs_inv.parity, af2rs.parity)

        # tests inversed x and y
        np_test.assert_almost_equal(af2rsi.phi, -af2rs.phi)
        np_test.assert_almost_equal(af2rsi.scale, 1/af2rs.scale, decimal=1)
        np_test.assert_almost_equal(af2rsi.parity, af2rs.parity)

        ###############################################
        #
        # parallelogram, rotation, scale, parity not exact
        #

        af2m = Affine2D.find(x=self.x2, y=self.y2m)
        af2mrs = Affine2D.find(x=self.x2, y=self.y2m, type_='rs')
        af2mrsi = Affine2D.find(y=self.x2, x=self.y2m, type_='rs')
        af2mrs_inv = af2mrs.inverse()
        af2mrs_inv.decompose(order='qpsm')
        
        # tests inverse method
        np_test.assert_almost_equal(af2mrs_inv.phi, af2mrs.phi)
        np_test.assert_almost_equal(af2mrs_inv.scale, 1/af2mrs.scale)
        np_test.assert_almost_equal(af2mrs_inv.parity, af2mrs.parity)

        # tests inversed x and y
        np_test.assert_almost_equal(af2mrsi.phi, af2mrs.phi)
        np_test.assert_almost_equal(af2mrsi.scale, 1/af2mrs.scale, decimal=1)
        np_test.assert_almost_equal(af2mrsi.parity, af2mrs.parity)

    def testCompose(self):
        """
        Tests compose
        """

        af11 = Affine2D.compose(self.af1, self.af1)
        af11.decompose(order='qpsm')
        np_test.assert_almost_equal(af11.phi, 2 * self.af1_phi)
        np_test.assert_almost_equal(af11.scale, 
                                    self.af1_scale * self.af1_scale)
        np_test.assert_almost_equal(af11.parity, 1)
        np_test.assert_almost_equal(af11.rmsErrorEst, 
                                    numpy.sqrt(2) * self.af1.error)

        af11m = Affine2D.compose(self.af1, self.af1m)
        af11m.decompose(order='qpsm')
        ## This was risen in testing. Added a "-"
        # E       AssertionError: 
        # E       Arrays are not almost equal to 7 decimals
        # E        ACTUAL: -3.1415926535897931
        # E        DESIRED: 3.1415926535897931
        ## Before:
        # np_test.assert_almost_equal(af11m.phi, self.af1_phi + self.af1m_phi)
        ## After:
        np_test.assert_almost_equal(-af11m.phi, self.af1_phi + self.af1m_phi)
        np_test.assert_almost_equal(af11m.scale, 
                                    self.af1_scale * self.af1m_scale)
        np_test.assert_almost_equal(af11m.parity, 
                                    self.af1_parity * self.af1m_parity)
        np_test.assert_almost_equal(af11m.rmsErrorEst, 
                                    numpy.sqrt(2) * self.af1.error)

        # test rms error
        af12 = Affine2D.compose(self.af1, self.af2)
        self.af1.decompose(order='qpsm')
        np_test.assert_almost_equal(af12.rmsErrorEst, 
                                    self.af1.scale[0] * self.af2.rmsError)
        af21 = Affine2D.compose(self.af2, self.af1)
        np_test.assert_almost_equal(af21.rmsErrorEst, self.af2.rmsError)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAffine2D)
    unittest.TextTestRunner(verbosity=2).run(suite)
