"""
Contains class Affine2D for preforming affine transformation (general linear
transformation followed by translation) on points (vectors) in 2D.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: affine_2d.py 1311 2016-06-13 12:41:50Z vladan $
"""

__version__ = "$Revision: 1311 $"


import logging
import numpy
import scipy
import scipy.linalg as linalg

from .points import Points
from .affine import Affine

class Affine2D(Affine):
    """
    Finds and preforms affine transformation (general linear transformation
    followed by translation) on points (vectors) in 2D.

    The transformation that transforms points x to points y has the following 
    form:

      y = gl x + d

    where:

      gl = q s p m

    Main methods:

      - find: finds a transformation between two sets of points
      - transform: transforms a (set of) point(s)
      - inverse: calculates inverse transformation
      - compose: composition of two transformations

    Important attributes and properties (see formulas above):

      - d: translation vector
      - gl: general linear transformation matrix
      - q: rotation matrix
      - phi: rotational angle (radians)
      - phiDeg: rotational angle (degrees)
      - s: scaling matrix (diagonal)
      - scale: vector of scaling parameters (diagonal of s)
      - p: parity matrix (diagonal)
      - parity: parity (+1 or -1)
      - m: shear matrix (upper-triangular)
      - error: error of transformation for all points
      - rmsError: root mean square error of the transformation
    """

    def __init__(self, d=None, gl=None, phi=None, scale=None, 
                 parity=1, shear=0, order='qpsm', xy_axes='point_dim'):
        """
        Initialization. Following argument combinations are valid:
          - no arguments: no transformation matrices are initialized
          - d and gl: d and gl are set
          - d, phi and scale (parity and shear optional): d and gl 
          (gl = q p s m) are set

        If arg d is None it is set to [0, 0]. If it is a single number it is 
        set to the same value in both directions.
          
        If the arg xy_axes is 'point_dim' / 'dim_point', points used in this 
        instance should be specified as n_point x 2 / 2 x n_point 
        matrices.
        
         Arguments
          - gl: gl matrix
          - phi: angle
          - scale: single number or 1d array
          - parity: 1 or -1
          - shear: (number) shear
          - d: single number, 1d array, or None (means 0)
          - order: decomposition order
          - xy_axes: order of axes in matrices representing points, can be
          'point_dim' (default) or 'dim_point'
        """

        # set d
        if d is None:
            d = 0
        if not isinstance(d, (numpy.ndarray, list)):
            d = self.makeD(d)

        if (gl is not None) and (d is not None):
            super(self.__class__, self).__init__(
                gl, d, order=order, xy_axes=xy_axes)

        elif (phi is not None) and (scale is not None) and (d is not None):
            
            if not isinstance(scale, (numpy.ndarray, list)):
                scale = self.makeS(scale)
            elif isinstance(scale, numpy.adarray) and (len(scale.shape) == 1):
                scale = self.makeS(scale)
            elif isinstance(scale, list) and not isinstance(scale[0], list):
                scale = self.makeS(scale)

            qp = numpy.inner(self.makeQ(phi), self.makeP(parity))
            sm = numpy.inner(scale, self.makeM(shear))
            gl = numpy.inner(qp, sm)
            super(self.__class__, self).__init__(
                gl, d, order=order, xy_axes=xy_axes)

        else:

            raise ValueError("Transformation could not be created because "
                             + " not enough parameters were specified.")

    @classmethod
    def downcast(cls, affine):
        """
        Returns instance of this class that was obtained by dowoncasting
        art affine (instance of Affine, base class of this class).
        
        Argument:
          - affine: instance of Affine
        """

        # copy gl and d, obligatory
        new = cls(gl=affine.gl, d=affine.d, xy_axes=affine.xy_axes)

        # copy attributes that are not obligarory
        for name in ['order', 'resids', 'rank', 'singular', 'error', '_xPrime', 
                     '_yPrime', 'q', 'p', 's', 'm', 'xy_axes']:
            try:
                value = getattr(affine, name)
                setattr(new, name, value)
            except AttributeError:
                pass
                
        return new

    ##############################################################
    #
    # Parameters
    #

    @classmethod
    def identity(cls, ndim=2):
        """
        Returnes an identity object of this class, that is a transformation 
        that leaves all vectors invariant.

        Argument ndim is ignored, it should be 2 here.
        """

        obj = cls.__base__.identity(ndim=2)        
        return obj

    @classmethod
    def makeQ(cls, phi):
        """
        Returns rotation matrix corresponding to angle phi
        """
        q = numpy.array([[numpy.cos(phi), -numpy.sin(phi)],
                         [numpy.sin(phi), numpy.cos(phi)]])
        return q

    @classmethod
    def getAngle(cls, q):
        """
        Returns angle corresponding to the rotation matrix specified by arg q
        """
        
        res = numpy.arctan2(q[1,0], q[0,0])
        return res

    @classmethod
    def makeS(cls, scale):
        """
        Returns scale transformation corresponding to 1D array scale.

        Argument:
          - scale: can be given as an 1d array (or a list), or as a single 
          number in which case the scale is the same in all directions
        """

        s = cls.__base__.makeS(scale=scale, ndim=2)
        return s

    @classmethod
    def makeP(cls, parity, axis=-1):
        """
        Returns parity matrix corresponding to arg parity. 

        If parity is -1, the element of the parity matrix corresponding to 
        axis is set to -1 (all other are 1).

        Arguments:
          - parity: can be 1 or -1
          - axis: axis denoting parity element that can be -1
        """

        p = cls.__base__.makeP(parity=parity, axis=axis, ndim=2)
        return p

    @classmethod
    def makeM(cls, shear):
        """
        Returns share matrix corresponding to (arg) shear.
        """
        m = numpy.array([[1, shear], 
                         [0, 1]])
        return m

    @classmethod
    def makeD(cls, d):
        """
        Returns d (translation) array corresponding to arg parity. 

        Arguments:
          - d: (single number) translation
        """

        d = cls.__base__.makeD(d, ndim=2)
        return d

    def getPhi(self):
        """
        Rotation angle of matrix self.q in radians.
        """
        #try:
        #    qq = self.q
        #except AttributeError:
        #    self.decompose(order='qpsm')
        res = numpy.arctan2(self.q[1,0], self.q[0,0])
        return res

    def setPhi(self, phi):
        """
        Sets transformation matrices related to phi (q and gl). Matrix gl is
        calculated using the current values of other matrices (p, s and m).
        """
        self.q = self.makeQ(phi)
        try:
            gg = self.gl
            self.gl = self.composeGl()
        except AttributeError:
            pass

    phi = property(fget=getPhi, fset=setPhi, doc='Rotation angle in radians')

    def getPhiDeg(self):
        """
        Rotation angle in degrees
        """
        res = self.phi * 180 / numpy.pi
        return res

    def setPhiDeg(self, phi):
        """
        Sets transformation matrices related to phi (q and gl). Matrix gl is
        calculated using the current values of other matrices (p, s and m).
        """
        phi_rad = phi * numpy.pi / 180
        self.q = self.makeQ(phi_rad)
        try:
            gg = self.gl
            self.gl = self.composeGl()
        except AttributeError:
            pass
    phiDeg = property(fget=getPhiDeg, fset=setPhiDeg, 
                      doc='Rotation angle in degrees')

    def getUAngle(self):
        """
        Returns angle alpha corresponding to rotation matrix self.u
        """
        return self.getAngle(q=self.u)

    def setUAngle(self, angle):
        """
        Sets U matrix (as in usv decomposition) and adjusts gl.
        """
        self.u = self.makeQ(angle)
        self.gl = self.composeGl()

    uAngle = property(fget=getUAngle, fset=setUAngle, 
                   doc='Rotation angle corresponding to matrix U in radians')

    def getUAngleDeg(self):
        """
        Returns angle alpha corresponding to rotation matrix self.u
        """
        res =  self.getAngle(q=self.u) * 180 / numpy.pi
        return res

    def setUAngleDeg(self, angle):
        """
        Sets U matrix (as in usv decomposition) and adjusts gl.
        """
        angle_rad = angle * numpy.pi / 180
        self.u = self.makeQ(angle_rad)
        self.gl = self.composeGl()

    uAngleDeg = property(fget=getUAngleDeg, fset=setUAngleDeg, 
                   doc='Rotation angle corresponding to matrix U in degrees')

    def getVAngle(self):
        """
        Returns angle alpha corresponding to rotation matrix self.v
        """
        return self.getAngle(q=self.v)

    def setVAngle(self, angle):
        """
        Sets V matrix (as in usv decomposition) and adjusts gl.
        """
        self.v = self.makeQ(angle)
        self.gl = self.composeGl()

    vAngle = property(fget=getVAngle, fset=setVAngle, 
                   doc='Rotation angle corresponding to matrix V in radians')

    def getVAngleDeg(self):
        """
        Returns angle alpha corresponding to rotation matrix self.v
        """
        res =  self.getAngle(q=self.v) * 180 / numpy.pi
        return res

    def setVAngleDeg(self, angle):
        """
        Sets V matrix (as in usv decomposition) and adjusts gl.
        """
        angle_rad = angle * numpy.pi / 180
        self.v = self.makeQ(angle_rad)
        self.gl = self.composeGl()

    vAngleDeg = property(fget=getVAngleDeg, fset=setVAngleDeg, 
                   doc='Rotation angle corresponding to matrix V in degrees')

    def getScaleAngle(self):
        """
        Returns angle (in rad) that corresponds to the scaling: 

          arccos(scale_smaller / scale_larger)

        where scale_smaller and scale_larger are the smaller and larger scale
        factors, respectively.

        Rotation of an 2D object by this angle around x-axis in 3D is eqivalent
        to scaling this object by self.scale (up to a common scale factor).  
        """
        ratio = self.scale[1] / self.scale[0]
        if ratio > 1:
            ratio = 1. / ratio
        res = numpy.arccos(ratio)
        return res

    scaleAngle = property(
        fget=getScaleAngle, 
        doc='Angle corresponding to the ratio of scales (in rad)')

    def getScaleAngleDeg(self):
        """
        Returns angle in degrees that corresponds to the scaling: 

          arccos(scale[1]/scale[0])

        Rotation of an 2D object by this angle around x-axis in 3D is eqivalent
        to scaling this object by self.scale (up to a common scale factor).  
        """
        return self.scaleAngle * 180 / numpy.pi

    scaleAngleDeg = property(
        fget=getScaleAngleDeg, 
        doc='Angle corresponding to the ratio of scales in degrees')

    def getShear(self):
        """
        Shear
        """
        try:
            mm = self.m
        except AttributeError:
            self.decompose()
        res = self.m[0, 1]
        return res

    shear = property(fget=getShear, doc='Shear')

    ##############################################################
    #
    # Finding and applying transformations
    #

    @classmethod
    def find(
            cls, x, y, x_ref='cm', y_ref='cm', type_='gl', xy_axes='point_dim'):
        """
        Finds affine transformation (general linear transformation folowed by a
        translation) that minimizes square error for transforming points x to 
        points y in 2D. The transformation has the form

          y = gl x + d,                                            (1)

        and:
 
          gl = q s p m   for type_='gl'
          gl = S q p     for type_='rs'
 
        where d is translation vector, q, s, p and m are rotation, scaling,
        parity and shear matrices, respectivly and S is a scalar scale (same
        for both directions)

        In the default mode (x_ref='cm' and y_ref='cm') the parameters are
        calculated by minimizing square error to get gl from:

          y - y_cm = gl (x - x_cm)   and   d = y_cm - gl x_cm

        where x_cm and y_cm are the centers of mass for x and y respectivly.
        In this case the square error of eq 1 is minimized

        In case args x_ref and y_ref are coordinates, gl is determined by
        minimizing square error in:

          y - y_ref = gl (x - x_ref)   and d = y_ref - gl x_ref

        Note that in this case the parameters found do not minimize the error
        of eq 1.

        In case type_='gl', general linear transformation (matrix gl) is
        calculated using Affine.find which in turn uses scipy.linalg.lstsq.

        Alternatively, if type_='rs', rotation angle parity and scale are
        calculated using findRS() method.

        Arguments:
          - x, y: sets of points, both having shape (n_points, n_dim)
          - x_ref, y_ref: (ndarray) coordinates of reference points, or 'cm' to
          use center of mass

        Returns the transformation found as an instance of class cls, with 
        following attributes:
          - gl: general linear transformation matrix
          - d: translation vector
          - q, p, s, m: rotation, parity, scale and shear matrices 
          - error: difference between y and transformed x values
          - resids, rank, singular: values returned from scipy.linalg.lstsq
          - _xPrime: x - x_ref
          - _yPrime: y - y_ref
          - type_: type of the optimization, 'gl' to find Gl transformation
          that optimizes the square error, or 'rs' to find the best rotation 
          and one scale (currently implemented for 2D transformations only)
        """

        if type_ == 'gl':

            # run Affine.base and downcast
            base_inst = cls.__base__.find(
                x=x, y=y, x_ref=x_ref, y_ref=y_ref, xy_axes=xy_axes)
            inst = cls.downcast(affine=base_inst)

        elif type_ == 'rs':

            # call special method for 'rs' type in 2D
            inst = cls.findRS(
                x=x, y=y, x_ref=x_ref, y_ref=y_ref, xy_axes=xy_axes)

        else:
            raise ValueError("Argument type_: ", type_, "was not ",
                             "understood. Valid values are 'gl', and 'rs'.")

        return inst

    @classmethod
    def findRS(cls, x, y, x_ref='cm', y_ref='cm', xy_axes='point_dim'):
        """
        Finds transformation consisting of rotation, single scale factor and
        translation in 2D that minimizes square error for transforming points
        x to points y. The transformation has the form

          y = gl x + d,    gl = S q p                                     (1)
 
        where d is translation vector, q and p are rotation and parity 
        matrices, respectivly and S is a scalar scale (same for both 
        directions)

        In the default mode (x_ref='cm' and y_ref='cm') the parameters are
        calculated by minimizing square error to get gl from:

          y - y_cm = gl (x - x_cm)   and   d = y_cm - gl x_cm

        where x_cm and y_cm are the centers of mass for x and y respectivly.
        In this case the square error of eq 1 is minimized

        In case args x_ref and y_ref are coordinates, gl is determined by
        minimizing square error in:

          y - y_ref = gl (x - x_ref)   and d = y_ref - gl x_ref

        Note that in this case the parameters found do not minimize the error
        of eq 1.

        In center of mass coordinates, scale and parity are calculated 
        directly using:
        
          S = sqrt( det(yx) / det(xx) )

          P = sign( det(yx) / det(xx) )

        Rotation angle is calculated so that the square error is minimized:

          tan(phi + pi/2) = tr(y p x) / tr(y r0 p x)

        where:
          
          r0 = 0 -1
               1  0

        Arguments:
          - x, y: sets of points, both having shape (n_points, n_dim)
          - x_ref, y_ref: (ndarray) coordinates of reference points, or 'cm' to
          use center of mass

        Returns the transformation found as an instance of class cls, with 
        following attributes:
          - gl: general linear transformation matrix
          - d: translation vector
          - q, p, s, m: rotation, parity, scale and shear matrices 
          - error: difference between y and transformed x values
          - resids, rank, singular: values returned from scipy.linalg.lstsq

        Note: To be replaced by SVD based method
        """

        # bring x and y to n_points x n_dim shape 
        if xy_axes == 'point_dim':
            pass
        elif xy_axes == 'dim_point':
            x = x.transpose()
            y = y.transpose()
        else:
            raise ValueError(
                "Argument xy_axes was not understood. Possible values are: "
                + "'point_dim' and 'dim_point'.")

        # bring x to reference frame
        if isinstance(x_ref, str) and (x_ref == 'cm'):
            x_ref = numpy.mean(x, axis=0)
        elif isinstance(x_ref, (list, tuple, numpy.ndarray)):
            pass
        else:
            raise ValueError(\
                'Argument x_ref: ', x_ref, ' was not understood.',
                " Allowed values are None, 'cm', or an array.") 
        x_prime = x - x_ref

        # bring y to reference frame
        if isinstance(y_ref, str) and (y_ref == 'cm'):
            y_ref = numpy.mean(y, axis=0)
        elif isinstance(y_ref, (list, tuple, numpy.ndarray)):
            pass
        else:
            raise ValueError(\
                'Argument y_ref: ', y_ref, ' was not understood.',
                " Allowed values are None, 'cm', or an array.") 
        y_prime = y - y_ref

        # find parity and scale
        det_xy = linalg.det(numpy.dot(x_prime.transpose(), y_prime))
        det_xx = linalg.det(numpy.dot(x_prime.transpose(), x_prime))
        parity = numpy.sign(det_xy * det_xx)
        scale = numpy.sqrt(parity * det_xy / float(det_xx))
        p = numpy.array([[1, 0], [0, parity]])
        s = numpy.diag([scale, scale])

        # find phi
        px = numpy.inner(x_prime, p)
        ypx = (y_prime * px).sum()
        s2 = numpy.array([[0, -1], [1, 0]])
        ys2px = (numpy.dot(y_prime, s2) * px).sum()
        phi = numpy.arctan2(-ypx, float(ys2px)) + numpy.pi / 2

        # q (rotation matrix)
        q = numpy.array([[numpy.cos(phi), -numpy.sin(phi)],
                        [numpy.sin(phi), numpy.cos(phi)]])

        # check +pi ambiguity of phi
        yqpx = (numpy.dot(y_prime, q) * numpy.inner(x_prime, p)).sum()
        if yqpx < 0:
            phi += numpy.pi
            q = cls.getQ(phi)
 
        # get gl and d and instantiate
        gl = numpy.dot(numpy.dot(q, s), p)
        d = y_ref - numpy.inner(x_ref, gl)
        inst = cls(gl=gl, d=d)
        inst.xy_axes = xy_axes

        # get error
        inst.error = y - inst.transform(x, xy_axes='point_dim')
        if xy_axes == 'dim_point':
            inst.error = inst.error.transpose()
 
        # save transformations
        inst.q = q
        inst.s = s
        inst.p = p
        inst.m = numpy.identity(2)
        #inst.gl = gl
   
        return inst

    def findConformal(cls, x, y, x_mask=None, y_mask=None, d=None):
        """
        Work in progress
        
        Finds conformal transformation (global scaling and rotation folowed by a
        translation) that transforms points x to points y:

          y = s q x + d

        The scale and rotation angle are determined from averages of lengths and
        angles respectivly, of vectors formed by x and y in respect to their
        centers of mass.

        The translation is calculated translation from:

          d = y_cm - gl x_cm

        where x_cm and y_cm are x and y centers of mass.
    
        If d (translation) is given the transformation is determined using 
        given d.

        Only the points that are not masked neither in x_mask nor in y_mask are
        used.

        Arguments:
          - x, y: sets of points, both having shape (n_points, n_dim)
          - x_mask, y_masks: masked (not used) points, vectors of length 
          n_points
          - d: translation vector of length ndim

        Returns an instance of the transformation found with following 
        attributes:
          - gl: transformation matrix
          - d: translation vector
        """

        raise NotImplementedError("Sorry, this is still work in progress.")

        # remove masked points
        [x, y], mask = cls.removeMasked([x,y], [x_mask,y_mask])

        # deal with mode
        if d is None:

            # bring x and y to cm frame
            x_cm = numpy.mean(x, axis=0)
            x_0 = x - x_cm
            y_cm = numpy.mean(y, axis=0)
            y_0 = y - y_cm

        else:
            x_0 = x
            y_0 = y - d

        # find scale
        x_dist = Points(x_0).distance()
        y_dist = Points(y_0).distance()
        scale = (y_dist / x_dist).mean()

        # find rotation
        numpy.arctan2(y_0, x_0)

    def decompose(self, gl=None, order=None):
        """
        Decomposes gl using QR or singular value decomposition as follows:

          gl = q p s m (order 'qr' or 'qpsm') 
          gl = p s m q (order 'rq' or 'psmq')
          gl = u p s v (order 'usv')

        where:
          - q, u, v: rotation matrix (orthogonal, with det +1)
          - p: parity matrix (diagonal, the element self.parity_axis can be +1 
          or -1, other diagonal elements +1)
          - s: scale martix, diagonal and >=0
          - m: shear matrix, upper triangular, all diagonal elements 1

        The order is determined by agr oder. In this case self.order is set to
        (arg) order). Otherwise, if arg order is None, self.order is used.

        In case of the singular value decomposition (order='usv'), the angle
        corresponding to rotation matrix U is set to be between -pi/2 and pi/2.
        This is achieved by rotation of both U and V matrices by pi (if needed).

        Note: uses decompose() from super for everything except the adjustment 
        of U (and V).

        Arguments:
          - gl: (ndarray) general linear transformation, or self.gl if None
          - order: decomposition order 'qpsm' (same as 'qr'), 'psmq' (same as 
          'rq'), or 'usv'

        If arg gl is None, self.gl us used and the matrices resulting from the 
        decomposition are saved as the arguments of this instance:
          - self.q, self.p, self.s and self.m if order 'qpsm', 'qr', 'psmq' 
          or 'rq'
          - self.u, self.p, self.s, self.v if order 'usv'

        Returns only if gl is not None:
          - (q, p, s, m) if order 'qpsm', 'qr', 'psmq' or 'rq'
          - (u, p, s, v) if order 'usv'
        """

        # figure out type of return 
        if gl is None:
            new = False
        else:
            new = True

        # decompose
        decomp = super(self.__class__, self).decompose(gl=gl, order=order)

        if order == 'usv':

            # adjust u and v
            if decomp is None:

                # modify attributes of this instance
                self.adjustUV()

            else:

                # make another instance and modify attributes there
                local = self.__class__(order='usv')
                (local.u, local.p, local.s, local.v) = decomp
                local.adjustUV()

        else:

            # just return whatever super.decompose() did
            return decomp

    def adjustUV(self):
        """
        In case of the singular value decomposition (order='usv'), the angle
        corresponding to rotation matrix U is set to be between -pi/2 and pi/2.
        This is achieved by rotation of both U and V matrices by pi (if needed).

        If the angle corresponding to the rotation matrix U is already between
        -pi/2 and pi/2, doesn't do anything.
        """
        
        if (self.uAngle > numpy.pi / 2) or (self.uAngle < -numpy.pi / 2):

            # adjust u
            self.uAngle += numpy.pi

            # adjust v
            self.vAngle += numpy.pi

            # compose (should not decompose)
            self.gl = self.composeGl(order='usv')

