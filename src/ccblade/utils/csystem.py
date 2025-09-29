#!/usr/bin/env python
# encoding: utf-8
"""
csystem.py

Created by Andrew Ning on 2/21/2012.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np


class DirectionVector(object):
    """Handles rotation of direction vectors to appropriate coordinate systems.
    All angles must be in degrees.

    """

    def __init__(self, x, y, z):
        """3-Dimensional vector that depends on direction only (not position).

        Parameters
        ----------
        x : float or ndarray
            x-direction of vector(s)
        y : float or ndarray
            y-direction of vector(s)
        z : float or ndarray
            z-direction of vector(s)

        """

        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

    @classmethod
    def fromArray(cls, array):
        """initialize with NumPy array

        Parameters
        ----------
        array : ndarray
            construct DirectionVector using array of size 3

        """

        return cls(array[0], array[1], array[2])

    def toArray(self):
        """convert DirectionVector to NumPy array

        Returns
        -------
        array : ndarray
            NumPy array in order x, y, z containing DirectionVector data

        """

        return np.c_[self.x, self.y, self.z]

    def _rotateAboutZ(self, xstring, ystring, zstring, theta, thetaname, reverse=False):
        """
        x X y = z.  rotate c.s. about z, +theta
        all angles in degrees
        """

        thetaM = 1.0
        if reverse:
            thetaM = -1.0

        x = getattr(self, xstring)
        y = getattr(self, ystring)
        z = getattr(self, zstring)

        theta = np.radians(theta * thetaM)
        c = np.cos(theta)
        s = np.sin(theta)

        xnew = x * c + y * s
        ynew = -x * s + y * c
        znew = z

        return xnew, ynew, znew

    def windToInertial(self, beta):
        """Rotates from wind-aligned to inertial

        Parameters
        ----------
        beta : float (deg)
            wind angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the inertial coordinate system

        """
        xw, yw, zw = self._rotateAboutZ("x", "y", "z", beta, "beta", reverse=True)
        return DirectionVector(xw, yw, zw)

    def inertialToWind(self, beta):
        """Rotates from inertial to wind-aligned

        Parameters
        ----------
        beta : float (deg)
            wind angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the wind-aligned coordinate system

        """
        xw, yw, zw = self._rotateAboutZ("x", "y", "z", beta, "beta")
        return DirectionVector(xw, yw, zw)

    def yawToWind(self, Psi):
        """Rotates from yaw-aligned to wind-aligned

        Parameters
        ----------
        Psi : float (deg)
            yaw angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the wind-aligned coordinate system

        """
        xw, yw, zw = self._rotateAboutZ("x", "y", "z", Psi, "yaw", reverse=True)
        return DirectionVector(xw, yw, zw)

    def windToYaw(self, Psi):
        """Rotates from wind-aligned to yaw-aligned

        Parameters
        ----------
        Psi : float (deg)
            yaw angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the yaw-aligned coordinate system

        """
        xy, yy, zy = self._rotateAboutZ("x", "y", "z", Psi, "yaw")
        return DirectionVector(xy, yy, zy)

    def hubToYaw(self, Theta):
        """Rotates from hub-aligned to yaw-aligned

        Parameters
        ----------
        Theta : float (deg)
            tilt angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the yaw-aligned coordinate system

        """
        zy, xy, yy = self._rotateAboutZ("z", "x", "y", Theta, "tilt", reverse=True)
        return DirectionVector(xy, yy, zy)

    def yawToHub(self, Theta):
        """Rotates from yaw-aligned to hub-aligned

        Parameters
        ----------
        Theta : float (deg)
            tilt angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the hub-aligned coordinate system

        """
        zh, xh, yh = self._rotateAboutZ("z", "x", "y", Theta, "tilt")
        return DirectionVector(xh, yh, zh)

    def hubToAzimuth(self, Lambda):
        """Rotates from hub-aligned to azimuth-aligned

        Parameters
        ----------
        Lambda : float or ndarray (deg)
            azimuth angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the azimuth-aligned coordinate system

        """
        yz, zz, xz = self._rotateAboutZ("y", "z", "x", Lambda, "azimuth")
        return DirectionVector(xz, yz, zz)

    def azimuthToHub(self, Lambda):
        """Rotates from azimuth-aligned to hub-aligned

        Parameters
        ----------
        Lambda : float or ndarray (deg)
            azimuth angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the hub-aligned coordinate system

        """

        yh, zh, xh = self._rotateAboutZ("y", "z", "x", Lambda, "azimuth", reverse=True)
        return DirectionVector(xh, yh, zh)

    def azimuthToBlade(self, Phi):
        """Rotates from azimuth-aligned to blade-aligned

        Parameters
        ----------
        Phi : float (deg)
            precone angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the blade-aligned coordinate system

        """

        zb, xb, yb = self._rotateAboutZ("z", "x", "y", Phi, "precone", reverse=True)
        return DirectionVector(xb, yb, zb)

    def bladeToAzimuth(self, Phi):
        """Rotates from blade-aligned to azimuth-aligned

        Parameters
        ----------
        Phi : float (deg)
            precone angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the azimuth-aligned coordinate system


        """

        za, xa, ya = self._rotateAboutZ("z", "x", "y", Phi, "precone")
        return DirectionVector(xa, ya, za)

    def airfoilToBlade(self, theta):
        """Rotates from airfoil-aligned to blade-aligned

        Parameters
        ----------
        theta : float (deg)
            twist angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the blade-aligned coordinate system

        """

        xb, yb, zb = self._rotateAboutZ("x", "y", "z", theta, "theta")
        return DirectionVector(xb, yb, zb)

    def bladeToAirfoil(self, theta):
        """Rotates from blade-aligned to airfoil-aligned

        Parameters
        ----------
        theta : float (deg)
            twist angle

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the airfoil-aligned coordinate system

        """

        xa, ya, za = self._rotateAboutZ("x", "y", "z", theta, "theta", reverse=True)
        return DirectionVector(xa, ya, za)

    def airfoilToProfile(self):
        """Rotates from airfoil-aligned to profile

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the profile coordinate system

        """

        return DirectionVector(self.y, self.x, self.z)

    def profileToAirfoil(self):
        """Rotates from profile to airfoil-aligned

        Returns
        -------
        vector : DirectionVector
            a DirectionVector in the airfoil-aligned coordinate system

        """

        return DirectionVector(self.y, self.x, self.z)

    def cross(self, other):
        """cross product between two DirectionVectors

        Parameters
        ----------
        other : DirectionVector
            other vector to cross with

        Returns
        -------
        vector : DirectionVector
            vector = self X other

        """
        v1 = np.c_[self.x, self.y, self.z]
        v2 = np.c_[other.x, other.y, other.z]
        v = np.cross(v1, v2)

        if len(v.shape) > 1:
            return DirectionVector(v[:, 0], v[:, 1], v[:, 2])
        else:
            return DirectionVector(v[0], v[1], v[2])

    def __neg__(self):
        """negate direction vector"""

        return DirectionVector(-self.x, -self.y, -self.z)

    def __add__(self, other):
        """add two DirectionVector objects (v1 = v2 + v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return DirectionVector(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        """subtract DirectionVector objects (v1 = v2 - v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return DirectionVector(self.x - other, self.y - other, self.z - other)

    def __iadd__(self, other):
        """add DirectionVector object to self (v1 += v2)"""

        if isinstance(other, DirectionVector):
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            self.x += other
            self.y += other
            self.z += other

        return self

    def __isub__(self, other):
        """subract DirectionVector object from self (v1 -= v2)"""

        if isinstance(other, DirectionVector):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        else:
            self.x -= other
            self.y -= other
            self.z -= other
        return self

    def __mul__(self, other):
        """multiply vector times a scalar or element by element muiltiply times another vector (v1 = alpha * v2 or v1 = v2 * v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return DirectionVector(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        """divide vector by a scalar or element by element division with another vector (v1 = v2 / alpha or v1 = v2 / v3)"""

        if isinstance(other, DirectionVector):
            return DirectionVector(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return DirectionVector(
                self.x / float(other), self.y / float(other), self.z / float(other)
            )

    def __imul__(self, other):
        """multiply self times a scalar or element by element muiltiply times another vector (v1 *= alpha or v1 *= v2)"""

        if isinstance(other, DirectionVector):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
        else:
            self.x *= other
            self.y *= other
            self.z *= other
        return self

    def __str__(self):
        """print string representation"""

        return "{0}, {1}, {2}".format(self.x, self.y, self.z)
