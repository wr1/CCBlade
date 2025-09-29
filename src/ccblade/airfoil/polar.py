#!/usr/bin/env python
"""
polar.py

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


class Polar:
    """Polar class for airfoil coefficients."""

    def __init__(self, Re, alpha, cl, cd, cm):
        """Constructor for Polar."""
        self.Re = Re
        self.alpha = np.array(alpha)
        self.cl = np.array(cl)
        self.cd = np.array(cd)
        self.cm = np.array(cm)

    def blend(self, other_polar, blend_weight):
        """Blend with another polar."""
        # generate merged set of angles of attack - get unique values
        alpha = np.union1d(self.alpha, other_polar.alpha)

        # truncate (TODO: could also have option to just use one of the polars for values out of range)
        min_alpha = max(self.alpha.min(), other_polar.alpha.min())
        max_alpha = min(self.alpha.max(), other_polar.alpha.max())
        alpha = alpha[np.logical_and(alpha >= min_alpha, alpha <= max_alpha)]

        # interpolate to new alpha
        cl1 = np.interp(alpha, self.alpha, self.cl)
        cl2 = np.interp(alpha, other_polar.alpha, other_polar.cl)
        cd1 = np.interp(alpha, self.alpha, self.cd)
        cd2 = np.interp(alpha, other_polar.alpha, other_polar.cd)
        cm1 = np.interp(alpha, self.alpha, self.cm)
        cm2 = np.interp(alpha, other_polar.alpha, other_polar.cm)

        # linearly blend
        Re = self.Re + blend_weight * (other_polar.Re - self.Re)
        cl = cl1 + blend_weight * (cl2 - cl1)
        cd = cd1 + blend_weight * (cd2 - cd1)
        cm = cm1 + blend_weight * (cm2 - cm1)

        return type(self)(Re, alpha, cl, cd, cm)

    def correction3D(
        self,
        radial_position_ratio,
        chord_to_radius_ratio,
        tip_speed_ratio,
        alpha_max_corr=30,
        alpha_linear_min=-5,
        alpha_linear_max=5,
    ):
        """Apply 3-D corrections."""
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
        lam = tip_speed_ratio / (1 + tip_speed_ratio**2) ** 0.5  # modified tip speed ratio
        expon = d / lam / radial_position_ratio
        expon_d = d / lam / radial_position_ratio / 2.0

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min, alpha <= alpha_linear_max)
        p = np.polyfit(alpha[idx], cl_2d[idx], 1)
        m = p[0]
        alpha0 = -p[1] / m

        # correction factor
        fcl = 1.0 / m * (1.6 * chord_to_radius_ratio / 0.1267 * (a - chord_to_radius_ratio**expon) / (b + chord_to_radius_ratio**expon) - 1)
        fcd = 1.0 / m * (1.6 * chord_to_radius_ratio / 0.1267 * (a - chord_to_radius_ratio**expon_d) / (b + chord_to_radius_ratio**expon_d) - 1)

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

        return type(self)(self.Re, np.degrees(alpha), cl_3d, cd_3d, self.cm)

    def extrapolate(self, max_cd, aspect_ratio=None, min_cd=0.001, num_alpha_points=15):
        """Extrapolate to high angles."""
        if min_cd < 0:
            raise Exception("cdmin cannot be < 0")

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

        # estimate CD max
        if aspect_ratio is not None:
            max_cd = 1.11 + 0.018 * aspect_ratio
        self.cdmax = max(max(self.cd), max_cd)

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
        alpha1 = np.linspace(alpha_high, np.pi / 2, num_alpha_points)
        alpha1 = alpha1[1:]  # remove first element so as not to duplicate when concatenating
        cl1, cd1 = self.__Viterna(alpha1, 1.0)

        # 90 <-> 180-alpha_high
        alpha2 = np.linspace(np.pi / 2, np.pi - alpha_high, num_alpha_points)
        alpha2 = alpha2[1:]
        cl2, cd2 = self.__Viterna(np.pi - alpha2, -cl_adj)

        # 180-alpha_high <-> 180
        alpha3 = np.linspace(np.pi - alpha_high, np.pi, num_alpha_points)
        alpha3 = alpha3[1:]
        cl3, cd3 = self.__Viterna(np.pi - alpha3, 1.0)
        cl3 = (alpha3 - np.pi) / alpha_high * cl_high * cl_adj  # override with linear variation

        if alpha_low <= -alpha_high:
            alpha4 = []
            cl4 = []
            cd4 = []
            alpha5max = alpha_low
        else:
            # -alpha_high <-> alpha_low
            # Note: this is done slightly differently than AirfoilPrep for better continuity
            alpha4 = np.linspace(-alpha_high, alpha_low, num_alpha_points)
            alpha4 = alpha4[1:-2]  # also remove last element for concatenation for this case
            cl4 = -cl_high * cl_adj + (alpha4 + alpha_high) / (alpha_low + alpha_high) * (cl_low + cl_high * cl_adj)
            cd4 = cd_low + (alpha4 - alpha_low) / (-alpha_high - alpha_low) * (cd_high - cd_low)
            alpha5max = -alpha_high

        # -90 <-> -alpha_high
        alpha5 = np.linspace(-np.pi / 2, alpha5max, num_alpha_points)
        alpha5 = alpha5[1:]
        cl5, cd5 = self.__Viterna(-alpha5, -cl_adj)

        # -180+alpha_high <-> -90
        alpha6 = np.linspace(-np.pi + alpha_high, -np.pi / 2, num_alpha_points)
        alpha6 = alpha6[1:]
        cl6, cd6 = self.__Viterna(alpha6 + np.pi, cl_adj)

        # -180 <-> -180 + alpha_high
        alpha7 = np.linspace(-np.pi, -np.pi + alpha_high, num_alpha_points)
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

        cd = np.maximum(cd, min_cd)  # don't allow negative drag coefficients

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
            cl_cm = np.interp(alpha_cm, np.degrees(alpha), cl)  # get cl for applicable alphas
            cd_cm = np.interp(alpha_cm, np.degrees(alpha), cd)  # get cd for applicable alphas
            alpha_low_deg = self.alpha[0]
            alpha_high_deg = self.alpha[-1]
            for i in range(len(alpha_cm)):
                cm_new = self.__getCM(i, cmCoef, alpha_cm, cl_cm, cd_cm, alpha_low_deg, alpha_high_deg)
                if cm_new is None:
                    pass  # For when it reaches the range of cm's that the user provides
                else:
                    cm_ext[i] = cm_new
        cm = np.interp(np.degrees(alpha), alpha_cm, cm_ext)
        return type(self)(self.Re, np.degrees(alpha), cl, cd, cm)

    def __Viterna(self, alpha, cl_adjustment):
        """Viterna extrapolation."""
        alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

        cl = self.cdmax / 2 * np.sin(2 * alpha) + self.A * np.cos(alpha) ** 2 / np.sin(alpha)
        cl = cl * cl_adjustment

        cd = self.cdmax * np.sin(alpha) ** 2 + self.B * np.cos(alpha)

        return cl, cd

    def __CMCoeff(self, cl_high, cd_high, cm_high):
        """Get CM coefficient."""
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
        XM = (-cm_high + cm0) / (cl_high * np.cos(alpha_high) + cd_high * np.sin(alpha_high))
        cmCoef = (XM - 0.25) / np.tan(alpha_high - np.pi / 2)
        return cmCoef

    def __getCM(self, i, cmCoef, alpha, cl_ext, cd_ext, alpha_low_deg, alpha_high_deg):
        """Extrapolate CM."""
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
                        cl_ext[i] * np.cos(np.radians(alpha[i])) + cd_ext[i] * np.sin(np.radians(alpha[i]))
                    )
                else:
                    x = cmCoef * np.tan(-np.radians(alpha[i]) - np.pi / 2) + 0.25
                    cm_new = -(
                        self.cm0
                        - x * (-cl_ext[i] * np.cos(-np.radians(alpha[i])) + cd_ext[i] * np.sin(-np.radians(alpha[i])))
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
                print("Angle encountered for which there is no CM table value (near +/-180 deg). Program will stop.")
        return cm_new

    def unsteadyparam(self, alpha_linear_min=-5, alpha_linear_max=5):
        """Compute unsteady parameters."""
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
        """Plot polar."""
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
