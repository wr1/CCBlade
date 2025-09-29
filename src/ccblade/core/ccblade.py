#!/usr/bin/env python
"""
CCBlade: Blade element momentum aerodynamics for wind turbines
"""

import logging
import warnings

import numpy as np
from scipy.optimize import brentq

from .bem import *
from .datamodels import BladeLoads, RotorOutputs

logger = logging.getLogger(__name__)


class CCBlade:
    """CCBlade class for blade element momentum analysis."""

    def __init__(
        self,
        r,
        chord,
        theta,
        af,
        Rhub,
        Rtip,
        B=3,
        rho=1.225,
        mu=1.81206e-5,
        precone=0.0,
        tilt=0.0,
        yaw=0.0,
        shearExp=0.2,
        hubHt=80.0,
        nSector=8,
        precurve=None,
        precurveTip=0.0,
        presweep=None,
        presweepTip=0.0,
        tiploss=True,
        hubloss=True,
        wakerotation=True,
        usecd=True,
        iterRe=1,
    ):
        """Constructor for aerodynamic rotor analysis."""
        r = np.array(r)
        self.r = r.copy()
        self.chord = np.array(chord)
        self.theta = np.deg2rad(theta)
        self.af = af
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.B = B
        self.rho = rho
        self.mu = mu
        self.precone = np.deg2rad(precone).item()
        self.tilt = np.deg2rad(tilt).item()
        self.yaw = np.deg2rad(yaw).item()
        self.shearExp = shearExp
        self.hubHt = hubHt
        self.bemoptions = {"usecd": usecd, "tiploss": tiploss, "hubloss": hubloss, "wakerotation": wakerotation}
        self.iterRe = iterRe

        # check if no precurve / presweep
        if precurve is None:
            precurve = np.zeros(len(r))
            precurveTip = 0.0

        if presweep is None:
            presweep = np.zeros(len(r))
            presweepTip = 0.0

        self.precurve = precurve.copy()
        self.precurveTip = precurveTip
        self.presweep = presweep.copy()
        self.presweepTip = presweepTip

        # Enfore unique points at tip and hub
        r_nd = (np.array(r) - r[0]) / (r[-1] - r[0])
        nd_hub = np.minimum(0.01, r_nd[:2].mean())
        nd_tip = np.maximum(0.99, r_nd[-2:].mean())
        if Rhub == r[0]:
            self.r[0] = np.interp(nd_hub, r_nd, r)
        if Rtip == r[-1]:
            self.r[-1] = np.interp(nd_tip, r_nd, r)
        if precurveTip == precurve[-1]:
            self.precurve[-1] = np.interp(nd_tip, r_nd, precurve)
        if presweepTip == presweep[-1]:
            self.presweep[-1] = np.interp(nd_tip, r_nd, presweep)

        # # rotor radius
        # if self.precurveTip != 0 and self.precone != 0.0:
        # print('rotor diameter may be modified in unexpected ways if tip precurve and precone are both nonzero')

        self.rotorR = Rtip * np.cos(self.precone) + self.precurveTip * np.sin(self.precone)

        # azimuthal discretization
        if self.tilt == 0.0 and self.yaw == 0.0 and self.shearExp == 0.0:
            self.nSector = 1  # no more are necessary
        else:
            self.nSector = max(4, nSector)  # at least 4 are necessary

        self.inverse_analysis = False
        self.induction = False
        self.induction_inflow = False

    # residual
    def __runBEM(self, phi, r, chord, theta, af, Vx, Vy):
        """Run BEM iteration."""
        axial_induction = 0.0
        tangential_induction = 0.0

        for _i in range(self.iterRe):
            alpha, W, Re = relativewind(
                phi, axial_induction, tangential_induction, Vx, Vy, self.pitch, chord, theta, self.rho, self.mu
            )
            cl, cd = af.evaluate(alpha, Re)

            fzero, axial_induction, tangential_induction = inductionfactors(
                r,
                chord,
                self.Rhub,
                self.Rtip,
                phi,
                cl,
                cd,
                self.B,
                Vx,
                Vy,
                **self.bemoptions,
            )

        return fzero, axial_induction, tangential_induction, cl, cd

    def __errorFunction(self, phi, r, chord, theta, af, Vx, Vy):
        """Error function for BEM."""
        fzero, axial_induction, tangential_induction, _, _ = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)
        return fzero

    def __runBEM_inverse(self, phi, r, chord, cl, cd, af, Vx, Vy):
        """Run inverse BEM."""
        axial_induction = 0.0
        tangential_induction = 0.0
        for _i in range(self.iterRe):
            fzero, axial_induction, tangential_induction = inductionfactors(
                r,
                chord,
                self.Rhub,
                self.Rtip,
                phi,
                cl,
                cd,
                self.B,
                Vx,
                Vy,
                **self.bemoptions,
            )

        return fzero, axial_induction, tangential_induction

    def __errorFunction_inverse(self, phi, r, chord, cl, cd, af, Vx, Vy):
        """Error function for inverse BEM."""
        fzero, axial_induction, tangential_induction = self.__runBEM_inverse(phi, r, chord, cl, cd, af, Vx, Vy)
        return fzero

    def __loads(self, phi, rotating, r, chord, theta, af, Vx, Vy):
        """Compute loads at section."""
        if Vx != 0.0 and Vy != 0.0:
            cphi = np.cos(phi)
            sphi = np.sin(phi)

            if rotating:
                _, axial_induction, tangential_induction, cl, cd = self.__runBEM(phi, r, chord, theta, af, Vx, Vy)
            else:
                axial_induction = 0.0
                tangential_induction = 0.0

            alpha_rad, W, Re = relativewind(
                phi, axial_induction, tangential_induction, Vx, Vy, self.pitch, chord, theta, self.rho, self.mu
            )
            if not rotating:
                cl, cd = af.evaluate(alpha_rad, Re)

            cn = cl * cphi + cd * sphi  # these expressions should always contain drag
            ct = cl * sphi - cd * cphi

            q = 0.5 * self.rho * W**2
            Np = cn * q * chord
            Tp = ct * q * chord

            alpha_deg = np.rad2deg(alpha_rad)

            return (
                axial_induction,
                tangential_induction,
                Np,
                Tp,
                alpha_deg,
                cl,
                cd,
                cn,
                ct,
                q,
                W,
                Re,
            )

        else:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

    def __windComponents(self, Uinf, Omega, azimuth):
        """Compute wind components."""
        Vx, Vy = windcomponents(
            len(self.r),
            self.r,
            self.precurve,
            self.presweep,
            self.precone,
            self.yaw,
            self.tilt,
            azimuth,
            Uinf,
            Omega,
            self.hubHt,
            self.shearExp,
        )

        return Vx, Vy

    def distributedAeroLoads(self, Uinf, Omega, pitch, azimuth):
        """Compute distributed aerodynamic loads."""
        self.pitch = np.deg2rad(pitch)
        azimuth = np.rad2deg(azimuth)

        # component of velocity at each radial station
        Vx, Vy = self.__windComponents(Uinf, Omega, azimuth)

        # initialize
        n = len(self.r)
        axial_induction = np.zeros(n)
        tangential_induction = np.zeros(n)
        alpha = np.zeros(n)
        cl = np.zeros(n)
        cd = np.zeros(n)
        Np = np.zeros(n)
        Tp = np.zeros(n)
        cn = np.zeros(n)
        ct = np.zeros(n)
        q = np.zeros(n)
        Re = np.zeros(n)
        W = np.zeros(n)

        if self.inverse_analysis:
            errf = self.__errorFunction_inverse
            self.theta = np.zeros_like(self.r)
        else:
            errf = self.__errorFunction
        rotating = Omega != 0.0

        # ---------------- loop across blade ------------------
        for i in range(n):
            # index dependent arguments
            if self.inverse_analysis:
                args = (
                    self.r[i],
                    self.chord[i],
                    self.cl[i],
                    self.cd[i],
                    self.af[i],
                    Vx[i],
                    Vy[i],
                )
            else:
                args = (
                    self.r[i],
                    self.chord[i],
                    self.theta[i],
                    self.af[i],
                    Vx[i],
                    Vy[i],
                )

            if not rotating:  # non-rotating
                phi_star = np.pi / 2.0

            else:
                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------

                # set standard limits
                epsilon = 1e-6
                phi_lower = epsilon
                phi_upper = np.pi / 2

                if errf(phi_lower, *args) * errf(phi_upper, *args) > 0:  # an uncommon but possible case
                    if errf(-np.pi / 4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -np.pi / 4
                        phi_upper = -epsilon
                    else:
                        phi_lower = np.pi / 2
                        phi_upper = np.pi - epsilon

                try:
                    phi_star = brentq(errf, phi_lower, phi_upper, args=args)

                except ValueError:
                    warnings.warn("error.  check input values.", stacklevel=2)
                    phi_star = 0.0

                # ----------------------------------------------------------------

            if self.inverse_analysis:
                self.theta[i] = phi_star - self.alpha[i] - self.pitch  # rad
                args = (
                    self.r[i],
                    self.chord[i],
                    self.theta[i],
                    self.af[i],
                    Vx[i],
                    Vy[i],
                )

            # derivatives of residual

            (
                axial_induction[i],
                tangential_induction[i],
                Np[i],
                Tp[i],
                alpha[i],
                cl[i],
                cd[i],
                cn[i],
                ct[i],
                q[i],
                W[i],
                Re[i],
            ) = self.__loads(phi_star, rotating, *args)

            if np.isnan(Np[i]):
                logger.warning(f"NaNs at {i}/{n}: {phi_lower} {phi_star} {phi_upper}")
                axial_induction[i] = 0.0
                tangential_induction[i] = 0.0
                Np[i] = 0.0
                Tp[i] = 0.0
                alpha[i] = 0.0
                # print('warning, BEM convergence error, setting Np[%d] = Tp[%d] = 0.' % (i,i))

        loads = {
            "Np": Np,
            "Tp": Tp,
            "a": axial_induction,
            "ap": tangential_induction,
            "alpha": alpha,
            "Cl": cl,
            "Cd": cd,
            "Cn": cn,
            "Ct": ct,
            "W": W,
            "Re": Re,
        }

        return BladeLoads(**loads)

    def evaluate(self, Uinf, Omega, pitch, coefficients=False):
        """Run aerodynamic analysis."""
        # rename
        args = (
            self.r,
            self.precurve,
            self.presweep,
            self.precone,
            self.Rhub,
            self.Rtip,
            self.precurveTip,
            self.presweepTip,
        )
        nsec = self.nSector

        # initialize
        Uinf = np.array(Uinf).flatten()
        Omega = np.array(Omega).flatten()
        pitch = np.array(pitch).flatten()

        npts = len(Uinf)
        T = np.zeros(npts)
        Y = np.zeros(npts)
        Z = np.zeros(npts)
        Q = np.zeros(npts)
        My = np.zeros(npts)
        Mz = np.zeros(npts)
        Mb = np.zeros(npts)
        P = np.zeros(npts)

        azimuth_angles = np.linspace(0.0, 2 * np.pi, nsec + 1)[:-1]
        for i in range(npts):  # iterate across conditions
            for azimuth in azimuth_angles:  # integrate across azimuth
                ca = np.cos(azimuth)
                sa = np.sin(azimuth)

                # contribution from this azimuthal location
                loads = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], np.rad2deg(azimuth))
                Np, Tp, W = (loads.Np, loads.Tp, loads.W)

                Tsub, Ysub, Zsub, Qsub, Msub = thrusttorque(len(self.r), Np, Tp, *args)

                # Scale rotor quantities (thrust & torque) by num blades.  Keep blade root moment as is
                T[i] += self.B * Tsub / nsec
                Y[i] += self.B * (Ysub * ca - Zsub * sa) / nsec
                Z[i] += self.B * (Zsub * ca + Ysub * sa) / nsec
                Q[i] += self.B * Qsub / nsec
                My[i] += self.B * Msub * ca / nsec
                Mz[i] += self.B * Msub * sa / nsec
                Mb[i] += Msub / nsec

        # Power
        P = Q * Omega * np.pi / 30.0  # RPM to rad/s

        # normalize if necessary
        if coefficients:
            q = 0.5 * self.rho * Uinf**2
            A = np.pi * self.rotorR**2
            CP = P / (q * A * Uinf)
            CT = T / (q * A)
            CY = Y / (q * A)
            CZ = Z / (q * A)
            CQ = Q / (q * self.rotorR * A)
            CMy = My / (q * self.rotorR * A)
            CMz = Mz / (q * self.rotorR * A)
            CMb = Mb / (q * self.rotorR * A)

        outputs = {}

        outputs["P"] = P
        outputs["T"] = T
        outputs["Y"] = Y
        outputs["Z"] = Z
        outputs["Q"] = Q
        outputs["My"] = My
        outputs["Mz"] = Mz
        outputs["Mb"] = Mb
        outputs["W"] = W
        if coefficients:
            outputs["CP"] = CP
            outputs["CT"] = CT
            outputs["CY"] = CY
            outputs["CZ"] = CZ
            outputs["CQ"] = CQ
            outputs["CMy"] = CMy
            outputs["CMz"] = CMz
            outputs["CMb"] = CMb

        return RotorOutputs(**outputs)
