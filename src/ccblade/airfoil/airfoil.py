#!/usr/bin/env python
# encoding: utf-8
"""
airfoil.py

Created by Andrew Ning on 2012-04-16.
Copyright (c) NREL. All rights reserved.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
import copy

from scipy.interpolate import RectBivariateSpline, bisplev


class Polar(object):
    """Defines section lift, drag, and pitching moment coefficients as a
    function of angle of attack at a particular Reynolds number.

    """

    def __init__(self, Re, alpha, cl, cd, cm):
        """Constructor

        Parameters
        ----------
        Re : float
            Reynolds number
        alpha : ndarray (deg)
            angle of attack
        cl : ndarray
            lift coefficient
        cd : ndarray
            drag coefficient
        cm : ndarray
            moment coefficient
        """

        self.Re = Re
        self.alpha = np.array(alpha)
        self.cl = np.array(cl)
        self.cd = np.array(cd)
        self.cm = np.array(cm)

    def blend(self, other, weight):
        """Blend this polar with another one with the specified weighting

        Parameters
        ----------
        other : Polar
            another Polar object to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        polar : Polar
            a blended Polar

        """

        # generate merged set of angles of attack - get unique values
        alpha = np.union1d(self.alpha, other.alpha)

        # truncate (TODO: could also have option to just use one of the polars for values out of range)
        min_alpha = max(self.alpha.min(), other.alpha.min())
        max_alpha = min(self.alpha.max(), other.alpha.max())
        alpha = alpha[np.logical_and(alpha >= min_alpha, alpha <= max_alpha)]
        # alpha = np.array([a for a in alpha if a >= min_alpha and a <= max_alpha])

        # interpolate to new alpha
        cl1 = np.interp(alpha, self.alpha, self.cl)
        cl2 = np.interp(alpha, other.alpha, other.cl)
        cd1 = np.interp(alpha, self.alpha, self.cd)
        cd2 = np.interp(alpha, other.alpha, other.cd)
        cm1 = np.interp(alpha, self.alpha, self.cm)
        cm2 = np.interp(alpha, other.alpha, other.cm)

        # linearly blend
        Re = self.Re + weight * (other.Re - self.Re)
        cl = cl1 + weight * (cl2 - cl1)
        cd = cd1 + weight * (cd2 - cd1)
        cm = cm1 + weight * (cm2 - cm1)

        return type(self)(Re, alpha, cl, cd, cm)

    def correction3D(
        self,
        r_over_R,
        chord_over_r,
        tsr,
        alpha_max_corr=30,
        alpha_linear_min=-5,
        alpha_linear_max=5,
    ):
        """Applies 3-D corrections for rotating sections from the 2-D data.

        Parameters
        ----------
        r_over_R : float
            local radial position / rotor radius
        chord_over_r : float
            local chord length / local radial location
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        polar : Polar
            A new Polar object corrected for 3-D effects

        Notes
        -----
        The Du-Selig method :cite:`Du1998A-3-D-stall-del` is used to correct lift, and
        the Eggers method :cite:`Eggers-Jr2003An-assessment-o` is used to correct drag.


        """

        # rename and convert units for convenience
        alpha = np.radians(self.alpha)
        cl_2d = self.cl
        cd_2d = self.cd
        alpha_max_corr = np.radians(alpha_max_corr)
        alpha_linear_min = np.radians(alpha_linear_min)
        alpha_linear_max = np.radians(alpha_linear_max)

        # parameters in Du-Selig model
        a = 1
        b = 1
        d = 1
        lam = tsr / (1 + tsr**2) ** 0.5  # modified tip speed ratio
        expon = d / lam / r_over_R
        expon_d = d / lam / r_over_R / 2.0

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min, alpha <= alpha_linear_max)
        p = np.polyfit(alpha[idx], cl_2d[idx], 1)
        m = p[0]
        alpha0 = -p[1] / m

        # correction factor
        fcl = (
            1.0
            / m
            * (
                1.6
                * chord_over_r
                / 0.1267
                * (a - chord_over_r**expon)
                / (b + chord_over_r**expon)
                - 1
            )
        )
        fcd = (
            1.0
            / m
            * (
                1.6
                * chord_over_r
                / 0.1267
                * (a - chord_over_r**expon_d)
                / (b + chord_over_r**expon_d)
                - 1
            )
        )

        # not sure where this adjustment comes from (besides AirfoilPrep spreadsheet of course)
        adj = ((np.pi / 2 - alpha) / (np.pi / 2 - alpha_max_corr)) ** 2
        adj[alpha <= alpha_max_corr] = 1.0

        # Du-Selig correction for lift
        cl_linear = m * (alpha - alpha0)
        cl_3d = cl_2d + fcl * (cl_linear - cl_2d) * adj

        # Du-Selig correction for drag
        cd0 = np.interp(0.0, alpha, cd_2d)
        dcd = cd_2d - cd0
        cd_3d = cd_2d + fcd * dcd

        # # Eggers 2003 correction for drag
        # delta_cl = cl_3d-cl_2d

        # delta_cd = delta_cl*(np.sin(alpha) - 0.12*np.cos(alpha))/(np.cos(alpha) + 0.12*np.sin(alpha))
        # cd_3d2 = cd_2d + delta_cd

        return type(self)(self.Re, np.degrees(alpha), cl_3d, cd_3d, self.cm)

    def extrapolate(self, cdmax, AR=None, cdmin=0.001, nalpha=15):
        """Extrapolates force coefficients up to +/- 180 degrees using Viterna's method
        :cite:`Viterna1982Theoretical-and`.

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            aspect ratio = (rotor radius / chord_75% radius)
            if provided, cdmax is computed from AR
        cdmin: float, optional
            minimum drag coefficient.  used to prevent negative values that can sometimes occur
            with this extrapolation method
        nalpha: int, optional
            number of points to add in each segment of Viterna method

        Returns
        -------
        polar : Polar
            a new Polar object

        Notes
        -----
        If the current polar already supplies data beyond 90 degrees then
        this method cannot be used in its current form and will just return itself.

        If AR is provided, then the maximum drag coefficient is estimated as

        >>> cdmax = 1.11 + 0.018*AR


        """

        if cdmin < 0:
            raise Exception("cdmin cannot be < 0")

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

        # estimate CD max
        if AR is not None:
            cdmax = 1.11 + 0.018 * AR
        self.cdmax = max(max(self.cd), cdmax)

        # extract matching info from ends
        alpha_high = np.radians(self.alpha[-1])
        cl_high = self.cl[-1]
        cd_high = self.cd[-1]
        cm_high = self.cm[-1]

        alpha_low = np.radians(self.alpha[0])
        cl_low = self.cl[0]
        cd_low = self.cd[0]

        if alpha_high > np.pi / 2:
            raise Exception("alpha[-1] > pi/2")
            return self
        if alpha_low < -np.pi / 2:
            raise Exception("alpha[0] < -pi/2")
            return self

        # parameters used in model
        sa = np.sin(alpha_high)
        ca = np.cos(alpha_high)
        self.A = (cl_high - self.cdmax * sa * ca) * sa / ca**2
        self.B = (cd_high - self.cdmax * sa * sa) / ca

        # alpha_high <-> 90
        alpha1 = np.linspace(alpha_high, np.pi / 2, nalpha)
        alpha1 = alpha1[
            1:
        ]  # remove first element so as not to duplicate when concatenating
        cl1, cd1 = self.__Viterna(alpha1, 1.0)

        # 90 <-> 180-alpha_high
        alpha2 = np.linspace(np.pi / 2, np.pi - alpha_high, nalpha)
        alpha2 = alpha2[1:]
        cl2, cd2 = self.__Viterna(np.pi - alpha2, -cl_adj)

        # 180-alpha_high <-> 180
        alpha3 = np.linspace(np.pi - alpha_high, np.pi, nalpha)
        alpha3 = alpha3[1:]
        cl3, cd3 = self.__Viterna(np.pi - alpha3, 1.0)
        cl3 = (
            (alpha3 - np.pi) / alpha_high * cl_high * cl_adj
        )  # override with linear variation

        if alpha_low <= -alpha_high:
            alpha4 = []
            cl4 = []
            cd4 = []
            alpha5max = alpha_low
        else:
            # -alpha_high <-> alpha_low
            # Note: this is done slightly differently than AirfoilPrep for better continuity
            alpha4 = np.linspace(-alpha_high, alpha_low, nalpha)
            alpha4 = alpha4[
                1:-2
            ]  # also remove last element for concatenation for this case
            cl4 = -cl_high * cl_adj + (alpha4 + alpha_high) / (
                alpha_low + alpha_high
            ) * (cl_low + cl_high * cl_adj)
            cd4 = cd_low + (alpha4 - alpha_low) / (-alpha_high - alpha_low) * (
                cd_high - cd_low
            )
            alpha5max = -alpha_high

        # -90 <-> -alpha_high
        alpha5 = np.linspace(-np.pi / 2, alpha5max, nalpha)
        alpha5 = alpha5[1:]
        cl5, cd5 = self.__Viterna(-alpha5, -cl_adj)

        # -180+alpha_high <-> -90
        alpha6 = np.linspace(-np.pi + alpha_high, -np.pi / 2, nalpha)
        alpha6 = alpha6[1:]
        cl6, cd6 = self.__Viterna(alpha6 + np.pi, cl_adj)

        # -180 <-> -180 + alpha_high
        alpha7 = np.linspace(-np.pi, -np.pi + alpha_high, nalpha)
        cl7, cd7 = self.__Viterna(np.pi - alpha7, 1.0)
        cl7 = (alpha7 + np.pi) / alpha_high * cl_high * cl_adj  # linear variation

        alpha = np.concatenate(
            (
                alpha7,
                alpha6,
                alpha5,
                alpha4,
                np.radians(self.alpha),
                alpha1,
                alpha2,
                alpha3,
            )
        )
        cl = np.concatenate((cl7, cl6, cl5, cl4, self.cl, cl1, cl2, cl3))
        cd = np.concatenate((cd7, cd6, cd5, cd4, self.cd, cd1, cd2, cd3))

        cd = np.maximum(cd, cdmin)  # don't allow negative drag coefficients

        # Setup alpha and cm to be used in extrapolation
        cm1_alpha = np.floor(self.alpha[0] / 10.0) * 10.0
        cm2_alpha = np.ceil(self.alpha[-1] / 10.0) * 10.0
        alpha_num = abs(int((-180.0 - cm1_alpha) / 10.0 - 1))
        alpha_cm1 = np.linspace(-180.0, cm1_alpha, alpha_num)
        alpha_cm2 = np.linspace(cm2_alpha, 180.0, int((180.0 - cm2_alpha) / 10.0 + 1))
        alpha_cm = np.concatenate(
            (alpha_cm1, self.alpha, alpha_cm2)
        )  # Specific alpha values are needed for cm function to work
        cm1 = np.zeros(len(alpha_cm1))
        cm2 = np.zeros(len(alpha_cm2))
        cm_ext = np.concatenate((cm1, self.cm, cm2))
        if np.count_nonzero(self.cm) > 0:
            cmCoef = self.__CMCoeff(cl_high, cd_high, cm_high)  # get cm coefficient
            cl_cm = np.interp(
                alpha_cm, np.degrees(alpha), cl
            )  # get cl for applicable alphas
            cd_cm = np.interp(
                alpha_cm, np.degrees(alpha), cd
            )  # get cd for applicable alphas
            alpha_low_deg = self.alpha[0]
            alpha_high_deg = self.alpha[-1]
            for i in range(len(alpha_cm)):
                cm_new = self.__getCM(
                    i, cmCoef, alpha_cm, cl_cm, cd_cm, alpha_low_deg, alpha_high_deg
                )
                if cm_new is None:
                    pass  # For when it reaches the range of cm's that the user provides
                else:
                    cm_ext[i] = cm_new
        cm = np.interp(np.degrees(alpha), alpha_cm, cm_ext)
        return type(self)(self.Re, np.degrees(alpha), cl, cd, cm)

    def __Viterna(self, alpha, cl_adj):
        """private method to perform Viterna extrapolation"""

        alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

        cl = self.cdmax / 2 * np.sin(2 * alpha) + self.A * np.cos(alpha) ** 2 / np.sin(
            alpha
        )
        cl = cl * cl_adj

        cd = self.cdmax * np.sin(alpha) ** 2 + self.B * np.cos(alpha)

        return cl, cd

    def __CMCoeff(self, cl_high, cd_high, cm_high):
        """private method to obtain CM0 and CMCoeff"""

        found_zero_lift = False

        for i in range(len(self.cm) - 1):
            if abs(self.alpha[i]) < 20.0 and self.cl[i] <= 0 and self.cl[i + 1] >= 0:
                p = -self.cl[i] / (self.cl[i + 1] - self.cl[i])
                cm0 = self.cm[i] + p * (self.cm[i + 1] - self.cm[i])
                found_zero_lift = True
                break

        if not found_zero_lift:
            p = -self.cl[0] / (self.cl[1] - self.cl[0])
            cm0 = self.cm[0] + p * (self.cm[1] - self.cm[0])
        self.cm0 = cm0
        alpha_high = np.radians(self.alpha[-1])
        XM = (-cm_high + cm0) / (
            cl_high * np.cos(alpha_high) + cd_high * np.sin(alpha_high)
        )
        cmCoef = (XM - 0.25) / np.tan((alpha_high - np.pi / 2))
        return cmCoef

    def __getCM(self, i, cmCoef, alpha, cl_ext, cd_ext, alpha_low_deg, alpha_high_deg):
        """private method to extrapolate Cm"""

        cm_new = 0
        if alpha[i] >= alpha_low_deg and alpha[i] <= alpha_high_deg:
            return
        if alpha[i] > -165 and alpha[i] < 165:
            if abs(alpha[i]) < 0.01:
                cm_new = self.cm0
            else:
                if alpha[i] > 0:
                    x = cmCoef * np.tan(np.radians(alpha[i]) - np.pi / 2) + 0.25
                    cm_new = self.cm0 - x * (
                        cl_ext[i] * np.cos(np.radians(alpha[i]))
                        + cd_ext[i] * np.sin(np.radians(alpha[i]))
                    )
                else:
                    x = cmCoef * np.tan(-np.radians(alpha[i]) - np.pi / 2) + 0.25
                    cm_new = -(
                        self.cm0
                        - x
                        * (
                            -cl_ext[i] * np.cos(-np.radians(alpha[i]))
                            + cd_ext[i] * np.sin(-np.radians(alpha[i]))
                        )
                    )
        else:
            if alpha[i] == 165:
                cm_new = -0.4
            elif alpha[i] == 170:
                cm_new = -0.5
            elif alpha[i] == 175:
                cm_new = -0.25
            elif alpha[i] == 180:
                cm_new = 0
            elif alpha[i] == -165:
                cm_new = 0.35
            elif alpha[i] == -170:
                cm_new = 0.4
            elif alpha[i] == -175:
                cm_new = 0.2
            elif alpha[i] == -180:
                cm_new = 0
            else:
                print(
                    "Angle encountered for which there is no CM table value "
                    "(near +/-180 deg). Program will stop."
                )
        return cm_new

    def unsteadyparam(self, alpha_linear_min=-5, alpha_linear_max=5):
        """compute unsteady aero parameters used in AeroDyn input file

        Parameters
        ----------
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        aerodynParam : tuple of floats
            (control setting, stall angle, alpha for 0 cn, cn slope,
            cn at stall+, cn at stall-, alpha for min CD, min(CD))

        """

        alpha = np.radians(self.alpha)
        cl = self.cl
        cd = self.cd

        alpha_linear_min = np.radians(alpha_linear_min)
        alpha_linear_max = np.radians(alpha_linear_max)

        cn = cl * np.cos(alpha) + cd * np.sin(alpha)

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min, alpha <= alpha_linear_max)

        # checks for inppropriate data (like cylinders)
        if len(idx) < 10 or len(np.unique(cl)) < 10:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        # linear fit
        p = np.polyfit(alpha[idx], cn[idx], 1)
        m = p[0]
        alpha0 = -p[1] / m

        # find cn at stall locations
        alphaUpper = np.radians(np.arange(40.0))
        alphaLower = np.radians(np.arange(5.0, -40.0, -1))
        cnUpper = np.interp(alphaUpper, alpha, cn)
        cnLower = np.interp(alphaLower, alpha, cn)
        cnLinearUpper = m * (alphaUpper - alpha0)
        cnLinearLower = m * (alphaLower - alpha0)
        deviation = 0.05  # threshold for cl in detecting stall

        alphaU = np.interp(deviation, cnLinearUpper - cnUpper, alphaUpper)
        alphaL = np.interp(deviation, cnLower - cnLinearLower, alphaLower)

        # compute cn at stall according to linear fit
        cnStallUpper = m * (alphaU - alpha0)
        cnStallLower = m * (alphaL - alpha0)

        # find min cd
        minIdx = cd.argmin()

        # return: control setting, stall angle, alpha for 0 cn, cn slope,
        #         cn at stall+, cn at stall-, alpha for min CD, min(CD)
        return (
            0.0,
            np.degrees(alphaU),
            np.degrees(alpha0),
            m,
            cnStallUpper,
            cnStallLower,
            alpha[minIdx],
            cd[minIdx],
        )

    def plot(self):
        """plot cl/cd/cm polar

        Returns
        -------
        figs : list of figure handles

        """
        import matplotlib.pyplot as plt

        p = self

        figs = []

        # plot cl
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        plt.plot(p.alpha, p.cl, label="Re = " + str(p.Re / 1e6) + " million")
        ax.set_xlabel("angle of attack (deg)")
        ax.set_ylabel("lift coefficient")
        ax.legend(loc="best")

        # plot cd
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cd, label="Re = " + str(p.Re / 1e6) + " million")
        ax.set_xlabel("angle of attack (deg)")
        ax.set_ylabel("drag coefficient")
        ax.legend(loc="best")

        # plot cm
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cm, label="Re = " + str(p.Re / 1e6) + " million")
        ax.set_xlabel("angle of attack (deg)")
        ax.set_ylabel("moment coefficient")
        ax.legend(loc="best")

        return figs


class Airfoil(object):
    """A collection of Polar objects at different Reynolds numbers"""

    def __init__(self, polars):
        """Constructor

        Parameters
        ----------
        polars : list(Polar)
            list of Polar objects

        """

        # sort by Reynolds number
        self.polars = sorted(polars, key=lambda p: p.Re)

        # save type of polar we are using
        self.polar_type = polars[0].__class__

    @classmethod
    def initFromAerodynFile(cls, aerodynFile, polarType=Polar):
        """Construct Airfoil object from AeroDyn file

        Parameters
        ----------
        aerodynFile : str
            path/name of a properly formatted Aerodyn file

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        # open aerodyn file
        f = open(aerodynFile, "r")

        # skip through header
        f.readline()
        description = f.readline().rstrip()  # remove newline
        f.readline()
        numTables = int(f.readline().split()[0])

        # loop through tables
        for i in range(numTables):
            # read Reynolds number
            Re = float(f.readline().split()[0]) * 1e6

            # read Aerodyn parameters
            param = [0] * 8
            for j in range(8):
                param[j] = float(f.readline().split()[0])

            alpha = []
            cl = []
            cd = []
            cm = []

            # read polar information line by line
            while True:
                line = f.readline()
                if "EOT" in line:
                    break
                data = [float(s) for s in line.split()]
                if len(data) < 4:
                    raise ValueError(
                        f"Error: Expected 4 columns of data but found, {data}"
                    )

                alpha.append(data[0])
                cl.append(data[1])
                cd.append(data[2])
                cm.append(data[3])

            polars.append(polarType(Re, alpha, cl, cd, cm))

        f.close()

        return cls(polars)

    def getPolar(self, Re):
        """Gets a Polar object for this airfoil at the specified Reynolds number.

        Parameters
        ----------
        Re : float
            Reynolds number

        Returns
        -------
        obj : Polar
            a Polar object

        Notes
        -----
        Interpolates as necessary. If Reynolds number is larger than or smaller than
        the stored Polars, it returns the Polar with the closest Reynolds number.

        """

        p = self.polars

        if Re <= p[0].Re:
            return copy.deepcopy(p[0])

        elif Re >= p[-1].Re:
            return copy.deepcopy(p[-1])

        else:
            Relist = [pp.Re for pp in p]
            i = np.searchsorted(Relist, Re)
            weight = (Re - Relist[i - 1]) / (Relist[i] - Relist[i - 1])
            return p[i - 1].blend(p[i], weight)

    def blend(self, other, weight):
        """Blend this Airfoil with another one with the specified weighting.


        Parameters
        ----------
        other : Airfoil
            other airfoil to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        obj : Airfoil
            a blended Airfoil object

        Notes
        -----
        First finds the unique Reynolds numbers.  Evaluates both sets of polars
        at each of the Reynolds numbers, then blends at each Reynolds number.

        """

        # combine Reynolds numbers
        Relist1 = [p.Re for p in self.polars]
        Relist2 = [p.Re for p in other.polars]
        Relist = np.union1d(Relist1, Relist2)

        # blend polars
        n = len(Relist)
        polars = [0] * n
        for i in range(n):
            p1 = self.getPolar(Relist[i])
            p2 = other.getPolar(Relist[i])
            polars[i] = p1.blend(p2, weight)

        return Airfoil(polars)

    def correction3D(
        self,
        r_over_R,
        chord_over_r,
        tsr,
        alpha_max_corr=30,
        alpha_linear_min=-5,
        alpha_linear_max=5,
    ):
        """apply 3-D rotational corrections to each polar in airfoil

        Parameters
        ----------
        r_over_R : float
            radial position / rotor radius
        chord_over_r : float
            local chord / local radius
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        airfoil : Airfoil
            airfoil with 3-D corrections

        See Also
        --------
        Polar.correction3D : apply 3-D corrections for a Polar

        """

        n = len(self.polars)
        polars = [0] * n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.correction3D(
                r_over_R,
                chord_over_r,
                tsr,
                alpha_max_corr,
                alpha_linear_min,
                alpha_linear_max,
            )

        return Airfoil(polars)

    def extrapolate(self, cdmax, AR=None, cdmin=0.001):
        """apply high alpha extensions to each polar in airfoil

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            blade aspect ratio (rotor radius / chord at 75% radius).  if included
            it is used to estimate cdmax
        cdmin: minimum drag coefficient

        Returns
        -------
        airfoil : Airfoil
            airfoil with +/-180 degree extensions

        See Also
        --------
        Polar.extrapolate : extrapolate a Polar to high angles of attack

        """

        n = len(self.polars)
        polars = [0] * n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.extrapolate(cdmax, AR, cdmin)

        return Airfoil(polars)

    def interpToCommonAlpha(self, alpha=None):
        """Interpolates all polars to a common set of angles of attack

        Parameters
        ----------
        alpha : ndarray, optional
            common set of angles of attack to use.  If None a union of
            all angles of attack in the polars is used.

        """

        if alpha is None:
            # union of angle of attacks
            alpha = []
            for p in self.polars:
                alpha = np.union1d(alpha, p.alpha)

        # interpolate each polar to new alpha
        n = len(self.polars)
        polars = [0] * n
        for idx, p in enumerate(self.polars):
            cl = np.interp(alpha, p.alpha, p.cl)
            cd = np.interp(alpha, p.alpha, p.cd)
            cm = np.interp(alpha, p.alpha, p.cm)
            polars[idx] = self.polar_type(p.Re, alpha, cl, cd, cm)

        return Airfoil(polars)

    def writeToAerodynFile(self, filename):
        """Write the airfoil section data to a file using AeroDyn input file style.

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        # aerodyn and wtperf require common set of angles of attack
        af = self.interpToCommonAlpha()

        f = open(filename, "w")

        f.write("AeroDyn airfoil file.")
        f.write("Compatible with AeroDyn v13.0.")
        f.write("Generated by airfoilprep.py")
        f.write(
            "{0:<10d}\t\t{1:40}".format(
                len(af.polars), "Number of airfoil tables in this file"
            )
        )
        for p in af.polars:
            f.write(
                "{0:<10f}\t{1:40}".format(p.Re / 1e6, "Reynolds number in millions.")
            )
            param = p.unsteadyparam()
            f.write("{0:<10f}\t{1:40}".format(param[0], "Control setting"))
            f.write("{0:<10f}\t{1:40}".format(param[1], "Stall angle (deg)"))
            f.write(
                "{0:<10f}\t{1:40}".format(
                    param[2], "Angle of attack for zero Cn for linear Cn curve (deg)"
                )
            )
            f.write(
                "{0:<10f}\t{1:40}".format(
                    param[3],
                    "Cn slope for zero lift for linear Cn curve (1/rad)"
                )
            )
            f.write(
                "{0:<10f}\t{1:40}".format(
                    param[4],
                    "Cn at stall value for positive angle of attack for linear Cn curve",
                )
            )
            f.write(
                "{0:<10f}\t{1:40}".format(
                    param[5],
                    "Cn at stall value for negative angle of attack for linear Cn curve",
                )
            )
            f.write(
                "{0:<10f}\t{1:40}".format(
                    param[6], "Angle of attack for minimum CD (deg)"
                )
            )
            f.write("{0:<10f}\t{1:40}".format(param[7], "Minimum CD value"))
            for a, cl, cd, cm in zip(p.alpha, p.cl, p.cd, p.cm):
                f.write("{:<10f}\t{:<10f}\t{:<10f}\t{:<10f}".format(a, cl, cd, cm))
            f.write("EOT")
        f.close()

    def createDataGrid(self):
        """interpolate airfoil data onto uniform alpha-Re grid.

        Returns
        -------
        alpha : ndarray (deg)
            a common set of angles of attack (union of all polars)
        Re : ndarray
            all Reynolds numbers defined in the polars
        cl : ndarray
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : ndarray
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """

        af = self.interpToCommonAlpha()
        polarList = af.polars

        # angle of attack is already same for each polar
        alpha = polarList[0].alpha

        # all Reynolds numbers
        Re = [p.Re for p in polarList]

        # fill in cl, cd grid
        cl = np.zeros((len(alpha), len(Re)))
        cd = np.zeros((len(alpha), len(Re)))
        cm = np.zeros((len(alpha), len(Re)))

        for idx, p in enumerate(polarList):
            cl[:, idx] = p.cl
            cd[:, idx] = p.cd
            cm[:, idx] = p.cm

        return alpha, Re, cl, cd, cm

    def plot(self, single_figure=True):
        """plot cl/cd/cm polars

        Parameters
        ----------
        single_figure : bool
            True  : plot all cl on the same figure (same for cd,cm)
            False : plot all cl/cd/cm on separate figures

        Returns
        -------
        figs : list of figure handles

        """

        import matplotlib.pyplot as plt

        figs = []

        # if in single figure mode (default)
        if single_figure:
            # generate figure handles
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            figs.append(fig1)

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            figs.append(fig2)

            # loop through all polars to see if we need to generate handles for cm figs
            for p in self.polars:
                if p.useCM == True:
                    fig3 = plt.figure()
                    ax3 = fig3.add_subplot(111)
                    figs.append(fig3)
                    break

            # loop through polars and plot
            for p in self.polars:
                # plot cl
                ax1.plot(p.alpha, p.cl, label="Re = " + str(p.Re / 1e6) + " million")
                ax1.set_xlabel("angle of attack (deg)")
                ax1.set_ylabel("lift coefficient")
                ax1.legend(loc="best")

                # plot cd
                ax2.plot(p.alpha, p.cd, label="Re = " + str(p.Re / 1e6) + " million")
                ax2.set_xlabel("angle of attack (deg)")
                ax2.set_ylabel("drag coefficient")
                ax2.legend(loc="best")

                # plot cm
                ax3.plot(p.alpha, p.cm, label="Re = " + str(p.Re / 1e6) + " million")
                ax3.set_xlabel("angle of attack (deg)")
                ax3.set_ylabel("moment coefficient")
                ax3.legend(loc="best")

        # otherwise, multi figure mode -- plot all on separate figures
        else:
            for p in self.polars:
                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cl, label="Re = " + str(p.Re / 1e6) + " million")
                ax.set_xlabel("angle of attack (deg)")
                ax.set_ylabel("lift coefficient")
                ax.legend(loc="best")

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cd, label="Re = " + str(p.Re / 1e6) + " million")
                ax.set_xlabel("angle of attack (deg)")
                ax.set_ylabel("drag coefficient")
                ax.legend(loc="best")

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cm, label="Re = " + str(p.Re / 1e6) + " million")
                ax.set_xlabel("angle of attack (deg)")
                ax.set_ylabel("moment coefficient")
                ax.legend(loc="best")

        return figs


# ------------------
#  CCAirfoil Class
# ------------------


class CCAirfoil(object):
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""

    def __init__(self, alpha, Re, cl, cd, cm=[], x=[], y=[], AFName="DEFAULTAF"):
        """Setup CCAirfoil from raw airfoil data on a grid.
        Parameters
        ----------
        alpha : array_like (deg)
            angles of attack where airfoil data are defined
            (should be defined from -180 to +180 degrees)
        Re : array_like
            Reynolds numbers where airfoil data are defined
            (can be empty or of length one if not Reynolds number dependent)
        cl : array_like
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        """

        alpha = np.deg2rad(alpha)
        self.x = x
        self.y = y
        self.AFName = AFName
        self.one_Re = False

        if len(cm) > 0:
            self.use_cm = True
        else:
            self.use_cm = False

        # special case if zero or one Reynolds number (need at least two for bivariate spline)
        if len(Re) < 2:
            Re = [1e1, 1e15]
            cl = np.c_[cl, cl]
            cd = np.c_[cd, cd]
            if self.use_cm:
                cm = np.c_[cm, cm]

            self.one_Re = True

        if len(alpha) < 2:
            raise ValueError(f"Need at least 2 angles of attack, but found {alpha}")

        kx = min(len(alpha) - 1, 3)
        ky = min(len(Re) - 1, 3)

        # a small amount of smoothing is used to prevent spurious multiple solutions
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.1)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)
        self.alpha = alpha

        if self.use_cm > 0:
            self.cm_spline = RectBivariateSpline(alpha, Re, cm, kx=kx, ky=ky, s=0.0001)

    @classmethod
    def initFromAerodynFile(cls, aerodynFile):
        """convenience method for initializing with AeroDyn formatted files
        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file
        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object
        """

        af = Airfoil.initFromAerodynFile(aerodynFile)
        alpha, Re, cl, cd, cm = af.createDataGrid()
        return cls(alpha, Re, cl, cd, cm=cm)

    def max_eff(self, Re):
        # Get the angle of attack, cl and cd at max airfoil efficiency. For a cylinder, set the angle of attack to 0

        Eff = np.zeros_like(self.alpha)

        # Check efficiency only between -20 and +40 deg
        aoa_start = -20.0
        aoa_end = 40
        i_start = np.argmin(abs(self.alpha - np.deg2rad(aoa_start)))
        i_end = np.argmin(abs(self.alpha - np.deg2rad(aoa_end)))

        if len(self.alpha[i_start:i_end]) == 0:  # Cylinder
            alpha_Emax = 0.0
            cl_Emax = self.cl_spline.ev(alpha_Emax, Re)
            cd_Emax = self.cd_spline.ev(alpha_Emax, Re)
            Emax = cl_Emax / cd_Emax

        else:
            alpha = np.deg2rad(np.linspace(aoa_start, aoa_end, num=201))
            cl = [self.cl_spline.ev(aoa, Re) for aoa in alpha]
            cd = [self.cd_spline.ev(aoa, Re) for aoa in alpha]
            Eff = [cli / cdi for cli, cdi in zip(cl, cd)]

            i_max = np.argmax(Eff)
            alpha_Emax = alpha[i_max]
            cl_Emax = cl[i_max]
            cd_Emax = cd[i_max]
            Emax = Eff[i_max]

        # print Emax, np.deg2rad(alpha_Emax), cl_Emax, cd_Emax

        return Emax, alpha_Emax, cl_Emax, cd_Emax

    def awayfromstall(self, Re, margin):
        # Get the angle of attack, cl and cd with a margin (in degrees) from the stall point. For a cylinder, set the angle of attack to 0 deg

        # Eff         = np.zeros_like(self.alpha)

        # Look for stall only between -20 and +40 deg
        aoa_start = -20.0
        aoa_end = 40
        i_start = np.argmin(abs(self.alpha - np.deg2rad(aoa_start)))
        i_end = np.argmin(abs(self.alpha - np.deg2rad(aoa_end)))

        if len(self.alpha[i_start:i_end]) == 0:  # Cylinder
            alpha_op = 0.0

        else:
            alpha = np.deg2rad(np.linspace(aoa_start, aoa_end, num=201))
            cl = [self.cl_spline.ev(aoa, Re) for aoa in alpha]
            cd = [self.cd_spline.ev(aoa, Re) for aoa in alpha]

            i_stall = np.argmax(cl)
            alpha_stall = alpha[i_stall]
            alpha_op = alpha_stall - np.deg2rad(margin)

        cl_op = self.cl_spline.ev(alpha_op, Re)
        cd_op = self.cd_spline.ev(alpha_op, Re)
        Eff_op = cl_op / cd_op

        # print Emax, np.deg2rad(alpha_Emax), cl_Emax, cd_Emax

        return Eff_op, alpha_op, cl_op, cd_op

    def evaluate(self, alpha, Re, return_cm=False):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.
        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.
        """

        cl = self.cl_spline.ev(alpha, Re)
        cd = self.cd_spline.ev(alpha, Re)

        if self.use_cm and return_cm:
            cm = self.cm_spline.ev(alpha, Re)
            return cl, cd, cm
        else:
            return cl, cd

    def derivatives(self, alpha, Re):
        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
        tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

        dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
        dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

        if self.one_Re:
            dcl_dRe = 0.0
            dcd_dRe = 0.0
        else:
            try:
                dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
                dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)
            except:
                dcl_dRe = 0.0
                dcd_dRe = 0.0
        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    def eval_unsteady(self, alpha, cl, cd, cm):
        # calculate unsteady coefficients from polars for OpenFAST's Aerodyn

        unsteady = {}

        alpha_rad = np.deg2rad(alpha)
        cn = cl * np.cos(alpha_rad) + cd * np.sin(alpha_rad)

        # alpha0, Cd0, Cm0
        aoa_l = [-30.0]
        aoa_h = [30.0]
        idx_low = np.argmin(abs(alpha - aoa_l))
        idx_high = np.argmin(abs(alpha - aoa_h))

        if max(np.abs(np.gradient(cl))) > 0.0:
            unsteady["alpha0"] = np.interp(
                0.0, cl[idx_low:idx_high], alpha[idx_low:idx_high]
            )
            unsteady["Cd0"] = np.interp(0.0, cl[idx_low:idx_high], cd[idx_low:idx_high])
            unsteady["Cm0"] = np.interp(0.0, cl[idx_low:idx_high], cm[idx_low:idx_high])
        else:
            unsteady["alpha0"] = 0.0
            unsteady["Cd0"] = cd[np.argmin(abs(alpha - 0.0))]
            unsteady["Cm0"] = 0.0

        unsteady["eta_e"] = 1
        unsteady["T_f0"] = "Default"
        unsteady["T_V0"] = "Default"
        unsteady["T_p"] = "Default"
        unsteady["T_VL"] = "Default"
        unsteady["b1"] = "Default"
        unsteady["b2"] = "Default"
        unsteady["b5"] = "Default"
        unsteady["A1"] = "Default"
        unsteady["A2"] = "Default"
        unsteady["A5"] = "Default"
        unsteady["S1"] = 0
        unsteady["S2"] = 0
        unsteady["S3"] = 0
        unsteady["S4"] = 0

        def find_breakpoint(x, y, idx_low, idx_high, multi=1.0):
            lin_fit = np.interp(
                x[idx_low:idx_high],
                [x[idx_low], x[idx_high]],
                [y[idx_low], y[idx_high]],
            )
            idx_break = 0
            lin_diff = 0
            for i, (fit, yi) in enumerate(zip(lin_fit, y[idx_low:idx_high])):
                if multi == 0:
                    diff_i = np.abs(yi - fit)
                else:
                    diff_i = multi * (yi - fit)
                if diff_i > lin_diff:
                    lin_diff = diff_i
                    idx_break = i
            idx_break += idx_low
            return idx_break

        # Cn1
        idx_alpha0 = np.argmin(abs(alpha - unsteady["alpha0"]))

        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            aoa_h = alpha[idx_alpha0] + 35.0
            idx_high = np.argmin(abs(alpha - aoa_h))

            cm_temp = cm[idx_low:idx_high]
            idx_cm_min = [
                i
                for i, local_min in enumerate(
                    np.r_[True, cm_temp[1:] < cm_temp[:-1]]
                    & np.r_[cm_temp[:-1] < cm_temp[1:], True]
                )
                if local_min
            ] + idx_low
            idx_high = idx_cm_min[-1]

            idx_Cn1 = find_breakpoint(alpha, cm, idx_alpha0, idx_high)
            unsteady["Cn1"] = cn[idx_Cn1]
        else:
            idx_Cn1 = np.argmin(abs(alpha - 0.0))
            unsteady["Cn1"] = 0.0

        # Cn2
        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            aoa_l = np.mean([alpha[idx_alpha0], alpha[idx_Cn1]]) - 30.0
            idx_low = np.argmin(abs(alpha - aoa_l))

            cm_temp = cm[idx_low:idx_high]
            idx_cm_min = [
                i
                for i, local_min in enumerate(
                    np.r_[True, cm_temp[1:] < cm_temp[:-1]]
                    & np.r_[cm_temp[:-1] < cm_temp[1:], True]
                )
                if local_min
            ] + idx_low
            idx_high = idx_cm_min[-1]

            idx_Cn2 = find_breakpoint(alpha, cm, idx_low, idx_alpha0, multi=0.0)
            unsteady["Cn2"] = cn[idx_Cn2]
        else:
            idx_Cn2 = np.argmin(abs(alpha - 0.0))
            unsteady["Cn2"] = 0.0

        # C_nalpha
        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            # unsteady['C_nalpha'] = np.gradient(cn, alpha_rad)[idx_alpha0]
            unsteady["C_nalpha"] = max(
                np.gradient(cn[idx_alpha0:idx_Cn1], alpha_rad[idx_alpha0:idx_Cn1])
            )

        else:
            unsteady["C_nalpha"] = 0.0

        # alpha1, alpha2
        # finding the break point in drag as a proxy for Trailing Edge separation, f=0.7
        # 3d stall corrections cause erroneous f calculations
        if max(np.abs(np.gradient(cm))) > 1.0e-10:
            aoa_l = [0.0]
            idx_low = np.argmin(abs(alpha - aoa_l))
            idx_alpha1 = find_breakpoint(alpha, cd, idx_low, idx_Cn1, multi=-1.0)
            unsteady["alpha1"] = alpha[idx_alpha1]
        else:
            idx_alpha1 = np.argmin(abs(alpha - 0.0))
            unsteady["alpha1"] = 0.0
        unsteady["alpha2"] = -1.0 * unsteady["alpha1"]

        unsteady["St_sh"] = "Default"
        unsteady["k0"] = 0
        unsteady["k1"] = 0
        unsteady["k2"] = 0
        unsteady["k3"] = 0
        unsteady["k1_hat"] = 0
        unsteady["x_cp_bar"] = "Default"
        unsteady["UACutout"] = "Default"
        unsteady["filtCutOff"] = "Default"

        unsteady["Alpha"] = alpha
        unsteady["Cl"] = cl
        unsteady["Cd"] = cd
        unsteady["Cm"] = cm

        self.unsteady = unsteady

    def af_flap_coords(
        self,
        xfoil_path,
        delta_flap=12.0,
        xc_hinge=0.8,
        yt_hinge=0.5,
        numNodes=250,
        multi_run=False,
        MPI_run=False,
    ):
        # This function is used to create and run xfoil to get airfoil coordinates for a given flap deflection
        # Set Needed parameter values
        AFName = self.AFName
        df = str(delta_flap)  # Flap deflection angle in deg
        numNodes = str(
            numNodes
        )  # number of panels to use (will be number of points in profile)
        dist_param = "0.5"  # TE/LE panel density ratio
        # Set filenames
        if multi_run or MPI_run:
            pid = mp.current_process().pid
            # Temporary file name for coordinates...will be deleted at the end
            CoordsFlnmAF = "profilecoords_p{}.dat".format(pid)
            saveFlnmAF = "{}_{}_Airfoil_p{}.txt".format(AFName, df, pid)
            saveFlnmPolar = "Polar_p{}.txt".format(pid)
            xfoilFlnm = "xfoil_input_p{}.txt".format(pid)
            NUL_fname = "NUL_{}".format(pid)
        else:
            CoordsFlnmAF = "profilecoords.dat"  # Temporary file name for coordinates...will be deleted at the end
            saveFlnmPolar = "Polar.txt"
            saveFlnmAF = "{}_{}_Airfoil.txt".format(AFName, df)
            xfoilFlnm = "xfoil_input.txt"  # Xfoil run script that will be deleted after it is no longer needed
            NUL_fname = "NUL"

        # Cleaning up old files to prevent replacement issues
        if os.path.exists(saveFlnmAF):
            os.remove(saveFlnmAF)
        if os.path.exists(CoordsFlnmAF):
            os.remove(CoordsFlnmAF)
        if os.path.exists(xfoilFlnm):
            os.remove(xfoilFlnm)
        if os.path.exists(NUL_fname):
            os.remove(NUL_fname)

        # Saving origional profile data temporatily to a txt file so xfoil can load it in
        dat = np.array([self.x, self.y])
        np.savetxt(CoordsFlnmAF, dat.T, fmt=["%f", "%f"])

        # %% Writes the Xfoil run script to read in coordinates, create flap, re-pannel, and save coordinates to a .txt file
        # Create the airfoil with flap
        fid = open(xfoilFlnm, "w")
        fid.write("PLOP \n G \n\n")  # turn off graphics
        fid.write("LOAD \n")
        fid.write(CoordsFlnmAF + "\n")  # name of file where coordinates are stored
        fid.write(AFName + "\n")  # name given to airfoil geometry (internal to xfoil)
        fid.write("GDES \n")  # enter geometric change options
        fid.write("FLAP \n")  # add in flap
        fid.write(str(xc_hinge) + "\n")  # specify x/c location of flap hinge
        fid.write("999\n")  # to specify y/t instead of actual distance
        fid.write(str(yt_hinge) + "\n")  # specify y/t value for flap hinge point
        fid.write(df + "\n")  # set flap deflection in deg
        fid.write("NAME \n")
        fid.write(AFName + "_" + df + "\n")  # name new airfoil
        fid.write("EXEC \n \n")  # move airfoil from buffer into current airfoil

        # Re-panel with specified number of panes and LE/TE panel density ratio (to possibly smooth out points)
        fid.write("PPAR\n")
        fid.write("N \n")
        fid.write(numNodes + "\n")
        fid.write("T \n")
        fid.write(dist_param + "\n")
        fid.write("\n\n")

        # Save airfoil coordinates with designation flap deflection
        fid.write("PSAV \n")
        fid.write(saveFlnmAF + " \n \n")  # the extra \n may not be needed

        # Quit xfoil and close xfoil script file
        fid.write("QUIT \n")
        fid.close()

        # Run the XFoil calling command
        os.system(
            xfoil_path + " < " + xfoilFlnm + " > " + NUL_fname
        )  # <<< runs XFoil !

        # Load in saved airfoil coordinates (with flap) from xfoil and save to instance variables
        flap_coords = np.loadtxt(saveFlnmAF)
        self.af_flap_xcoords = flap_coords[:, 0]
        self.af_flap_ycoords = flap_coords[:, 1]
        self.ctrl = delta_flap  # bem: the way that this function is called in rotor_geometry_yaml, this instance is not going to be used when calculating polars

        # Delete uneeded txt files script file
        if os.path.exists(CoordsFlnmAF):
            os.remove(CoordsFlnmAF)
        if os.path.exists(xfoilFlnm):
            os.remove(xfoilFlnm)
        if os.path.exists(saveFlnmAF):
            os.remove(saveFlnmAF)
        if os.path.exists(NUL_fname):
            os.remove(NUL_fname)
