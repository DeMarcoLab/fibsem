"""

Tests module rigid_3d

# Author: Vladan Lucic
# $Id: test_rigid_3d.py 1227 2015-08-25 10:39:27Z vladan $
"""

__version__ = "$Revision: 1227 $"

#from copy import copy, deepcopy
import unittest

import numpy as np
import numpy.testing as np_test
#import scipy as sp

from pyto.affine_2d import Affine2D
from pyto.rigid_3d import Rigid3D
#from rigid_3d import Rigid3D


class TestRigid3D(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.ninit = 10

    def test_find_32_constr_ck_scale_1(self):
        """
        Tests find_32_constr_ck(scale=1)
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # identity
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.gl, np.identity(3))
        np_test.assert_almost_equal(res.y, x_cs)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around z axis
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around z axis, 3 markers
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs[:,1:4], y=y[:2,1:4], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/6 rotation around z axis
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/6 rotation around z axis, 3 markers
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs[:,1:4], y=y[:2,1:4], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,1:4], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 8 pi/7 rotation around z axis
        r = Rigid3D.make_r_euler([0, 8*np.pi/7, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around x axis
        y = np.array([[0., 1, 0, 0],
                      [0, 0, 0, -3],
                      [0, 0, 2, 0]])
        r_desired = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/3 rotation around x axis
        r = Rigid3D.make_r_euler([np.pi/3, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/3 rotation around x axis, 3 markers
        r = Rigid3D.make_r_euler([np.pi/3, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs[:,1:4], y=y[:2,1:4], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,1:4], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 8 pi/9 rotation around x axis
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around y axis
        y = np.array([[0., 0, 0, 3],
                      [0, 0, 2, 0],
                      [0, -1, 0, 0]])
        r_desired = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r_desired, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/5 rotation around y axis
        r = Rigid3D.make_r_euler([np.pi/2, np.pi/5, -np.pi/2])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 7 pi/5 rotation around y axis
        r = Rigid3D.make_r_euler([np.pi/2, 7 * np.pi/5, -np.pi/2])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # identity, non-optimal initial
        # doesn't find optimal
        # cm=True improves but doesn't find optimal
        # fine if optimizing scale
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=1, cm=False, use_jac=False, 
            init=[0.2, -0.4, 0.5, -0.1])
        #np_test.assert_almost_equal(res.y, x_cs, decimal=3)
        #np_test.assert_almost_equal(res.gl, np.identity(3), decimal=3)
        #np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        # cm=True improves but doesn't find optimal
        # fine if optimizing scale
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=1, cm=False, use_jac=True, 
            init=[0.2, -0.4, 0.5, np.sqrt(0.55)])
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        # fine when +np.sqrt(0.55)
        # cm=True improves but doesn't find optimal
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=1, cm=False, use_jac=False, 
            init=[0.2, -0.4, 0.5, -np.sqrt(0.55)])

        # fails for 4, 5, 6 * pi/5, ok for 3 and 7
        # small init changes don't help
        # reducing z helps, the closer theta to pi the larger reduction  
        # cm=True improves but doesn't find optimal
        # fine if optimizing scale
        r = Rigid3D.make_r_euler([np.pi/2, 6 * np.pi/5, -np.pi/2])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True) 

        # pi around z (fi)
        # fails after 1 iter when init=[1, 0, 0, 0] and cm=False
        r = Rigid3D.make_r_euler([np.pi, 0, 0])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            init=[0.2, -0.4, 0.5, -np.sqrt(0.55)])
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi around z (psi)
        # fails after 1 iter when init=[1, 0, 0, 0] and cm=False
        r = Rigid3D.make_r_euler([0, 0, np.pi])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            init=[0.2, -0.4, 0.5, -np.sqrt(0.55)])
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example 1
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([50, 40, 24]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example 2
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([-150, 45, 130]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # Note: fails with default init when fi and psi interchanged, added to 
        # test_find_32_constr_ck_multi()
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([168, 32, -123]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example w noise
        x = np.array([[32., 78, 3, 41, 50, 47],
                      [13, 36, 54, 6, 38, 63],
                      [1.1, 3.5, 2.8, 4.2, 1.3, 3.2]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = np.dot(r, x_cm)
        y[:2,:] = y[:2,:] + np.array([[0.8, -0.3, 1.1, -0.9, 0.4, -0.5],
                                      [0.1, 0.9, 0.4, 0.1, -0.3, -0.4]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=y[:2,:], scale=1, cm=False, use_jac=True)
        #np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=2)
        #np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=2)
        #np_test.assert_almost_equal(res.gl, r, decimal=1)
        #np_test.assert_almost_equal(
        #    Rigid3D.extract_euler(res.gl, mode='x'), 
        #    angles, decimal=1)

    def test_find_32_constr_ck_scale_fixed(self):
        """
        Tests find_32_constr_ck() where scale is fixed but is not 1.
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # pi/2 rotation around z axis
        y = 3 * np.array([[0., 0, -2, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=3, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r_desired, decimal=3)
        np_test.assert_almost_equal(res.gl, 3 * r_desired, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 7 pi/5 rotation around y axis
        r = Rigid3D.make_r_euler([np.pi/2, 7 * np.pi/5, -np.pi/2])
        y = 4.5 * np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=4.5, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 4.5)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example
        # Note: fails with default init when fi and psi interchanged, added to 
        # test_find_32_constr_ck_multi()
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([168, 32, 123]) * np.pi / 180
        scale = 4.8
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale*np.dot(r, x_cm)[:2,:], scale=4.8, cm=False, 
            use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], 4.8 * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

    def test_find_32_constr_ck(self):
        """
        Tests find_32_constr_ck(scale=None) 
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # identity
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=None, cm=False, use_jac=False)
        np_test.assert_almost_equal(res.gl, np.identity(3))
        np_test.assert_almost_equal(res.y, x_cs)
        np_test.assert_almost_equal(res.s_scalar, 1)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi/2 rotation around z axis
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        r_desired = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True)
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r_desired, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 8 pi/9 rotation around x axis
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        scale = 12.3
        y = scale * np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(scale * np.dot(res.q, x_cs), y, decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # identity, non-optimal initial
        # doesn't find optimal without scale optimization
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=None, cm=False, use_jac=True, 
            init=[0.2, -0.4, 0.5, -0.1, 1])
        np_test.assert_almost_equal(np.dot(res.gl, x_cs), x_cs, decimal=3)
        np_test.assert_almost_equal(res.gl, np.identity(3), decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # identity, non-optimal initial
        # doesn't find optimal without scale optimization
        # still fails when -np.sqrt(0.55)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=x_cs[:2,:], scale=None, cm=False, use_jac=True, 
            init=[0.2, -0.4, 0.5, np.sqrt(0.55), 1])
        y = np.array([[0., 0, -2, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 3]])
        np_test.assert_almost_equal(res.y, x_cs, decimal=3)
        np_test.assert_almost_equal(res.q, np.identity(3), decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # 6 pi / 5 around y axis
        # fails without scale optimization
        r = Rigid3D.make_r_euler([np.pi/2, 6 * np.pi/5, -np.pi/2])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True) 
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi around z (fi)
        # fails with init=[1, 0, 0, 0], even if cm=True
        r = Rigid3D.make_r_euler([np.pi, 0, 0])
        y = 2.1 * np.dot(r, x_cs)
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=y[:2,:], scale=None, cm=False, use_jac=True,
            init=[0.9, 0.2, 0.2, 0.2, 1])
        np_test.assert_almost_equal(res.y[2,:], y[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 2.1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # pi around z (psi)
        # fails with init=[1, 0, 0, 0], even if cm=True
        r = Rigid3D.make_r_euler([0, 0, np.pi])
        res = Rigid3D.find_32_constr_ck(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=None, cm=False, use_jac=True,
            init=[0.9, 0.2, 0.2, 0.2, 1])
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example 1
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([50, 40, 24]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=None, cm=False, use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, 1, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.gl, mode='x'), 
            angles, decimal=3)

        # low z example 2
        x = np.array([[2., 5.3, 7.2, 0.3, 4],
                      [4, 3.2, 6, 5.4, 1.2],
                      [0.5, 1.2, 0.3, 0.5, 0.1]])
        angles = np.array([-150, 45, 130]) * np.pi / 180
        scale = 3.4
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # fails for default and some other init conditions
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([168, 32, 123]) * np.pi / 180
        scale = 1.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, init=[0.2, -0.4, 0.5, -np.sqrt(0.55), 1])
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # fails for some init conditions, even if init scale close to correct
        # for example on windows numpy/scipy with init=[-0.4, -0.3, 0.8, 0.2, 1] fails
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([134, 32, -78]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, init=[-0.4, -0.41, 0.8, 0.2, 1])
        if abs(res.optimizeResult.fun) > 0.001:
            print("\n"*2)
            print("="*50)
            print("OPTIMIZATION FAILED! Known problem on some platform. Please ignore!")
            print("="*50)
            print("\n"*2)
        else:
            np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
            np_test.assert_almost_equal(
                res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
            np_test.assert_almost_equal(res.q, r, decimal=3)
            np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
            np_test.assert_almost_equal(
                Rigid3D.extract_euler(res.q, mode='x'), 
                angles, decimal=3)

    def test_find_32_constr_ck_multi(self):
        """
        Tests find_32_constr_ck_multi()
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # theta = 6 pi/5, random init e
        r = Rigid3D.make_r_euler([0., 6 * np.pi/5, 0.])
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            ninit=self.ninit, randome=True, randoms=False) 
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # theta = 6 pi/5, phi = pi/2, theta = -pi/2, random init e
        r = Rigid3D.make_r_euler([np.pi/2, 6 * np.pi/5, -np.pi/2])
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            ninit=self.ninit, randome=True, randoms=False) 
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # theta = pi, phi = pi/2, theta = -pi/2 (identity), random init e
        r = Rigid3D.make_r_euler([np.pi/2, np.pi, -np.pi/2])
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, cm=False, use_jac=True,
            ninit=self.ninit, randome=True, randoms=False) 
        np_test.assert_almost_equal(res.y[2,:], np.dot(r, x_cs)[2,:], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # low z example 3, random init e, optimize scale
        # fails for some init conditions, even if init scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3, random init e, fixed scale
        # Note: fails with the default init e 
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=np.dot(r, x_cm)[:2,:], scale=1, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=False)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3, random init e and scale, optimize scale
        # fails for some init conditions (including default), even if init 
        # scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

        # low z example 3
        # fails for some init conditions (eg [0.4, -0.3, 0.8, 0.2, 1]), 
        # even if init scale close to correct
        x = np.array([[3.2, 7.8, 0.3, 4, 5],
                      [1.3, 3.6, 5.4, 6, 3.8],
                      [0.1, 0.5, 0.8, 0.2, 0.3]])
        angles = np.array([134, 32, -78]) * np.pi / 180
        scale = 56.
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        r = Rigid3D.make_r_euler(angles, mode='x')
        res = Rigid3D.find_32_constr_ck_multi(
            x=x_cm, y=scale * np.dot(r, x_cm)[:2,:], scale=None, cm=False, 
            use_jac=True, ninit=self.ninit, randome=True, randoms=True)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:], scale * np.dot(r, x_cm)[2,:], decimal=3)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, scale, decimal=3)
        np_test.assert_almost_equal(
            Rigid3D.extract_euler(res.q, mode='x'), 
            angles, decimal=3)

    def test_find_32(self):
        """
        Tests find_32()
        """

        # coord system-like points
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])

        # low z in cm frame
        x_low_z = np.array([[3.2, 7.8, 0.3, 4, 5],
                            [1.3, 3.6, 5.4, 6, 3.8],
                            [0.1, 0.5, 0.8, 0.2, 0.3]])
        x_low_z = x_low_z - x_low_z.mean(axis=-1).reshape((3,1))

        # transfom including translation, single default initial
        # pi/6 rotation around z axis
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        d = np.array([5,6,7.])
        y = np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=1, use_jac=True, ninit=1, randome=False,
            randoms=False)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:] - res.y[2:].mean(), 
            y[2,:] - y[2:].mean(), decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # transfom including translation and scale, single default initial
        # pi/6 rotation around z axis
        r = Rigid3D.make_r_euler([0, np.pi/6, 0])
        d = np.array([5,6,7.])
        s = 5.6
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, randome=False,
            randoms=False)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)
        np_test.assert_almost_equal(
            res.y[2,:] - res.y[2:].mean(), 
            y[2,:] - y[2:].mean(), decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.gl, s * r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)

        # single specified initial rotation, no scale nor translation
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        y = np.dot(r, x_cs)
        res = Rigid3D.find_32(
            x=x_cs, y=np.dot(r, x_cs)[:2,:], scale=1, use_jac=True, ninit=1, 
            randome=False, einit=[0.2, -0.4, 0.5, np.sqrt(0.55)], randoms=False)
        np_test.assert_almost_equal(res.gl, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # w translation and scale, single specified initial rotation
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        s = 124.3
        d = np.array([-3, -45., 17])
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit=[0.2, -0.4, 0.5, np.sqrt(0.55)], randoms=False)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # w translation and scale, single specified initial scale
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        s = 124.3
        d = np.array([-3, -45., 17])
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, randoms=False, sinit=0.5)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # w translation and scale, single specified initial rotation and scale
        r = Rigid3D.make_r_euler([8 * np.pi/9, 0, 0])
        s = 124.3
        d = np.array([-3, -45., 17])
        y = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_cs, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit=[0.2, -0.4, 0.5, np.sqrt(0.55)], 
            randoms=False, sinit=0.5)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.s_scalar, s, decimal=3)
        np_test.assert_almost_equal(res.d[:2], d[:2], decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # single specified initial rotation, w scale and translation
        # fails for default inits
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        #res = Rigid3D.find_32(
        #    x=x_low_z, y=y[:2,:], scale=None use_jac=True, ninit=1, 
        #    randome=False, randoms=False)
        #np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit=[0.6, -0.5, 0.3, np.sqrt(0.3)], randoms=False)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # single specified initial scale, w scale and translation
        # fails for default inits
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # single gl2 initial rotation, w scale and translation
        # fails for default inits
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=1, 
            randome=False, einit='gl2', randoms=False, sinit='gl2')
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation, single initial scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=None, randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial scale, single initial rotation
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=False, einit=None, randoms=True, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation and scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=None, randoms=True, sinit=None)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation around specified, single scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=[1,0,0,0], randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial scale around specified, single rotation
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=False, einit=[1,0,0,0], randoms=True, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation and scale around specified
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit=[1,0,0,0], randoms=True, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation around gl2, single scale
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit='gl2', randoms=False, sinit=0.2)
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial scale around gl2, single rotation
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=False, einit=[1,0,0,0], randoms=True, sinit='gl2')
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

        # random initial rotation and scale around gl2
        angles = np.array([-123, 32, 168]) * np.pi / 180
        s = 1
        d = np.array([3, -4., 5])
        r = Rigid3D.make_r_euler(angles, mode='x')
        y = s * np.dot(r, x_low_z) + np.expand_dims(d, 1)
        res = Rigid3D.find_32(
            x=x_low_z, y=y[:2,:], scale=None, use_jac=True, ninit=self.ninit, 
            randome=True, einit='gl2', randoms=True, sinit='gl2')
        np_test.assert_almost_equal(res.q, r, decimal=3)
        np_test.assert_almost_equal(res.optimizeResult.fun, 0, decimal=3)
        np_test.assert_almost_equal(res.y[:2,:], y[:2,:], decimal=3)

    def test_approx_gl2_to_ck3(self):
        """
        Test approx_gl2_to_ck3()
        """
        
        # coord system-like points
        x = np.array([[0., 1, 0, -1],
                      [0, 0, 2, -1],
                      [0, 0, 0, 0]])

        # arbitrary rotation, euler
        euler = np.array([60, 40, -35.]) * np.pi / 180
        e_euler = Rigid3D.euler_to_ck(euler)
        r = Rigid3D.make_r_euler(euler)
        y = Rigid3D().transform(x=x, q=r, s=1., d=0)
        e_res, s_res = Rigid3D().approx_gl2_to_ck3(x=x, y=y[:2,:], ret='both')
        np_test.assert_almost_equal(s_res, 1.)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_euler))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == np.sign(e_euler)).all() or
             (np.sign(e_res[0]) == -np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == -np.sign(e_euler)).all()), True)

        # arbitrary rotation, euler
        euler = np.array([100, 80, -170.]) * np.pi / 180
        e_euler = Rigid3D.euler_to_ck(euler)
        r = Rigid3D.make_r_euler(euler)
        y = Rigid3D().transform(x=x, q=r, s=35., d=0)
        e_res, s_res = Rigid3D().approx_gl2_to_ck3(x=x, y=y[:2,:], ret='both')
        np_test.assert_almost_equal(s_res, 35.)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_euler))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == np.sign(e_euler)).all() or
             (np.sign(e_res[0]) == -np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == -np.sign(e_euler)).all()), True)

        # arbitrary rotation, euler, theta > pi/2
        euler = np.array([100, 100, -170.]) * np.pi / 180
        euler_flipped = np.array([100, 80, -170.]) * np.pi / 180
        e_euler = Rigid3D.euler_to_ck(euler_flipped)
        r = Rigid3D.make_r_euler(euler)
        y = Rigid3D().transform(x=x, q=r, s=1., d=0)
        e_res, s_res = Rigid3D().approx_gl2_to_ck3(x=x, y=y[:2,:], ret='both')
        np_test.assert_almost_equal(s_res, 1.)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_euler))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == np.sign(e_euler)).all() or
             (np.sign(e_res[0]) == -np.sign(e_euler)).all() or
             (np.sign(e_res[1]) == -np.sign(e_euler)).all()), True)

        # arbitrary rotation, ck params, theta > pi/2
        e = np.array([-0.3, 0.7, 0.6, np.sqrt(0.06)])
        r = Rigid3D.make_r_ck(e=e)
        y = Rigid3D().transform(x=x, q=r, s=1, d=0)
        e_res, s_res = Rigid3D.approx_gl2_to_ck3(
            x=x, y=y[:2,:], xy_axes='dim_point')
        euler_flipped = Rigid3D.extract_euler(r, ret='one')
        euler_flipped[1] = np.pi - euler_flipped[1]
        e_flipped = Rigid3D.euler_to_ck(euler_flipped)
        np_test.assert_almost_equal(s_res, 1)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_flipped))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[0]) == -np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == -np.sign(e_flipped)).all()), True)

        # arbitrary rotation, ck params + scale, theta > pi/2
        e = np.array([-0.3, 0.7, 0.6, np.sqrt(0.06)])
        r = Rigid3D.make_r_ck(e=e)
        y = Rigid3D().transform(x=x, q=r, s=4.6, d=0)
        e_res, s_res = Rigid3D.approx_gl2_to_ck3(
            x=x, y=y[:2,:], xy_axes='dim_point')
        euler_flipped = Rigid3D.extract_euler(r, ret='one')
        euler_flipped[1] = np.pi - euler_flipped[1]
        e_flipped = Rigid3D.euler_to_ck(euler_flipped)
        np_test.assert_almost_equal(s_res, 4.6)
        np_test.assert_almost_equal(np.abs(e_res[0]), np.abs(e_flipped))
        np_test.assert_equal(
            ((np.sign(e_res[0]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == np.sign(e_flipped)).all() or
             (np.sign(e_res[0]) == -np.sign(e_flipped)).all() or
             (np.sign(e_res[1]) == -np.sign(e_flipped)).all()), True)

    def test_gl2_to_ck3(self):
        """
        Test gl2_to_ck3()
        """
        
        # make 3D r from known Gl and check angles
        euler = np.array([70, 50, -30.]) * np.pi / 180
        u = Affine2D.makeQ(euler[2])
        d = np.diag([1, np.cos(euler[1])])
        v = Affine2D.makeQ(euler[0])
        gl = 2.3 * np.dot(np.dot(u, d), v)
        res_e_param, res_s = Rigid3D.gl2_to_ck3(gl=gl, ret='one')
        res_r = Rigid3D.make_r_ck(res_e_param)
        res_euler = Rigid3D.extract_euler(res_r, mode='x', ret='one')
        np_test.assert_almost_equal(res_s, 2.3)
        np_test.assert_almost_equal(
            np.remainder(res_euler[0], np.pi), np.remainder(euler[0], np.pi))
        np_test.assert_almost_equal(np.abs(res_euler[1]), np.abs(euler[1]))
        np_test.assert_almost_equal(
            np.remainder(res_euler[2], np.pi), np.remainder(euler[2], np.pi))

    def test_make_r_ck(self):
        """
        Tests make_r_ck()
        """

        # invert angle
        res = np.dot(Rigid3D.make_r_ck([0.2, 0.4, 0.3, np.sqrt(0.71)]), 
                     Rigid3D.make_r_ck([-0.2, 0.4, 0.3, np.sqrt(0.71)]))
        np_test.assert_almost_equal(res, np.identity(3))
        np_test.assert_almost_equal(res, np.identity(3))

        # invert axis
        res = np.dot(Rigid3D.make_r_ck([0.5, 0.4, 0.6, np.sqrt(0.23)]), 
                     Rigid3D.make_r_ck([0.5, -0.4, -0.6, -np.sqrt(0.23)]))
        np_test.assert_almost_equal(res, np.identity(3))
        np_test.assert_almost_equal(res, np.identity(3))

    def test_make_r_euler(self):
        """
        Tests make_r_euler()
        """

        # arbitrary rotation and inverse
        res = np.dot(Rigid3D.make_r_euler([1., 2, 3]), 
                     Rigid3D.make_r_euler([-3., -2, -1]))
        np_test.assert_almost_equal(res, np.identity(3))

        # arbitrary rotation and inverse
        res = np.dot(Rigid3D.make_r_euler([-0.5, 1.2, 2.8]), 
                     Rigid3D.make_r_euler([-2.8, -1.2, 0.5]))
        np_test.assert_almost_equal(res, np.identity(3))

    def test_extract_euler(self):
        """
        Tests extract_euler()
        """

        # non-degenerate
        angles = [1.5, 0.7, -0.4]
        r = Rigid3D.make_r_euler(angles)
        res = Rigid3D.extract_euler(r, ret='both')
        angles_mod = np.remainder(angles, 2*np.pi)
        res_mod = np.remainder(res, 2*np.pi)
        np_test.assert_almost_equal(res_mod[0], angles_mod)
        desired = [angles[0] + np.pi, -angles[1], angles[2] + np.pi]
        desired_mod = np.remainder(desired, 2*np.pi)
        np_test.assert_almost_equal(res_mod[1], desired_mod)

        # degenerate
        angles = [1.5, 0., -0.4]
        r = Rigid3D.make_r_euler(angles)
        res = Rigid3D.extract_euler(r, ret='both')
        np_test.assert_almost_equal(res[0], [1.1, 0, 0])
        np_test.assert_almost_equal(res[1], [0, 0, 1.1])

    def test_euler_to_ck(self):
        """
        Tests euler_to_ck(). 

        Assumes make_r_euler() and make_r_ck() are correct.
        """

        angles = [-0.5, 1.2, 2.3]
        r_euler = Rigid3D.make_r_euler(angles, mode='x')
        e = Rigid3D.euler_to_ck(angles, mode='x')
        r_ck = Rigid3D.make_r_ck(e)
        np_test.assert_almost_equal(r_ck, r_euler)

    def test_make_random_ck(self):
        """
        Tests make_random_ck()
        """

        # repeat few times because random
        for ind in range(10):

            # no center
            res = Rigid3D.make_random_ck(center=None, distance=None)
            np_test.assert_almost_equal(np.square(res).sum(), 1)

            # center e_0 = 1
            distance = 0.2
            res = Rigid3D.make_random_ck(
                center=[1., 0, 0, 0], distance=distance)
            np_test.assert_almost_equal(np.square(res).sum(), 1)
            np_test.assert_equal((res[1:] <= distance).all(), True)

            # arbitrary center
            distance = 0.1
            center = [0.5, -0.7, 0.2, -np.sqrt(0.22)] 
            res = Rigid3D.make_random_ck(center=center, distance=distance)
            np_test.assert_almost_equal(np.square(res).sum(), 1)
            r_center = Rigid3D.make_r_ck(center)
            r_res = Rigid3D.make_r_ck(res)
            r_rt_res = np.dot(r_center.transpose(), r_res)
            rt_res_euler = Rigid3D.extract_euler(
                r=r_rt_res, ret='one', mode='x')
            rt_res = Rigid3D.euler_to_ck(rt_res_euler)
            np_test.assert_almost_equal(np.square(rt_res).sum(), 1)
            np_test.assert_equal((rt_res[1:] <= distance).all(), True)

    def test_transform(self):
        """
        Tests transform()
        """

        # coord system-like points and params
        x_cs = np.array([[0., 1, 0, 0],
                         [0, 0, 2, 0],
                         [0, 0, 0, 3]])
        angles = np.array([-123, 32, 168]) * np.pi / 180
        r = Rigid3D.make_r_euler(angles, mode='x')
        s = 23.
        d = [1., 2, 3]

        # dim_point
        rigid3d = Rigid3D()
        rigid3d.q = r
        rigid3d.s_scalar = s
        y_desired = s * np.dot(r, x_cs)
        y = rigid3d.transform(x=x_cs)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point
        rigid3d = Rigid3D()
        rigid3d.q = r
        rigid3d.s_scalar = s
        rigid3d.d = d
        y_desired = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        y = rigid3d.transform(x=x_cs)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point
        rigid3d = Rigid3D()
        rigid3d.s_scalar = s
        y_desired = s * np.dot(r, x_cs)
        y = rigid3d.transform(x=x_cs, q=r)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point
        rigid3d = Rigid3D()
        y_desired = s * np.dot(r, x_cs) + np.expand_dims(d, 1)
        y = rigid3d.transform(x=x_cs, q=r, s=s, d=d)
        np_test.assert_almost_equal(y, y_desired)
        
        # xy_axes=dim_point, xy_axes='point_dim'
        rigid3d = Rigid3D()
        rigid3d.q = r
        rigid3d.s_scalar = s
        rigid3d.d = d
        y_desired = s * np.inner(x_cs.transpose(), r) + d
        y = rigid3d.transform(x=x_cs.transpose(), xy_axes='point_dim')
        np_test.assert_almost_equal(y, y_desired)
        
        # d=None, xy_axes='point_dim'
        rigid3d = Rigid3D()
        y_desired = s * np.inner(x_cs.transpose(), r)
        y = rigid3d.transform(x=x_cs.transpose(), q=r, s=s, xy_axes='point_dim')
        np_test.assert_almost_equal(y, y_desired)
        
    def test_recalculate_translation(self):
        """
        Tests recalculate_translation()
        """

        # no initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        #r3d.d = np.array([1,2,0])
        center = np.array([0,1,0]).reshape(3,1)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-2, -2, 0]).reshape((3,1)))

        # with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        center = np.array([0,1,0]).reshape(3,1)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-1, 1, 1]).reshape((3,1)))

        # center point_dim form, with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([-1,3,1])
        center = np.array([0,1,0]).reshape(1,3)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-3, 1, 1]).reshape((1,3)))

        # center 1d form, with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 2.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        center = np.array([0,1,0]).reshape(1,3)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-1, 1, 1]).reshape((1,3)))

        # another example with initial translation
        r3d = Rigid3D()
        r3d.s_scalar = 3.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        center = np.array([2,0,0]).reshape(3,1)
        np_test.assert_almost_equal(
            r3d.recalculate_translation(rotation_center=center), 
            np.array([-5, 9, 1]).reshape((3,1)))

        # mimick rotation around another center
        r3d = Rigid3D()
        r3d.s_scalar = 3.
        angles = np.array([90, 0, 0]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        x=np.array([-1, 2, 1]).reshape((3,1))
        y_desired = r3d.transform(x=x)
        center = np.array([2,0,0]).reshape(3,1)
        y_actual = (
            r3d.s_scalar * (np.dot(r3d.q, x-center) + center) 
            + r3d.recalculate_translation(rotation_center=center))
        np_test.assert_almost_equal(y_actual, y_desired)
 
        # mimick rotation around another center, more complicated
        r3d = Rigid3D()
        r3d.s_scalar = 2.5
        angles = np.array([40, 67, -89]) * np.pi / 180
        r3d.q = Rigid3D.make_r_euler(angles, mode='x')
        r3d.d = np.array([1,3,1])
        x=np.array([-1, 2, 1]).reshape((3,1))
        y_desired = r3d.transform(x=x)
        center = np.array([3,2,1]).reshape(3,1)
        y_actual = (
            r3d.s_scalar * (np.dot(r3d.q, x-center) + center) 
            + r3d.recalculate_translation(rotation_center=center))
        np_test.assert_almost_equal(y_actual, y_desired)
 

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRigid3D)
    unittest.TextTestRunner(verbosity=2).run(suite)


