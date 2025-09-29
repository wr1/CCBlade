#!/usr/bin/env python
"""
ccairfoil.py

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

import multiprocessing as mp
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline, bisplev

from .airfoil import Airfoil


class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously differentiable cubic spline"""

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
                    np.r_[True, cm_temp[1:] < cm_temp[:-1]] & np.r_[cm_temp[:-1] < cm_temp[1:], True]
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
                    np.r_[True, cm_temp[1:] < cm_temp[:-1]] & np.r_[cm_temp[:-1] < cm_temp[1:], True]
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
