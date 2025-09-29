#!/usr/bin/env python
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

import copy
import numpy as np

from .polar import Polar


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
            "{0:<10d}		{1:40}".format(
                len(af.polars), "Number of airfoil tables in this file"
            )
        )
        for p in af.polars:
            f.write(
                "{0:<10f}	{1:40}".format(p.Re / 1e6, "Reynolds number in millions.")
            )
            param = p.unsteadyparam()
            f.write("{0:<10f}	{1:40}".format(param[0], "Control setting"))
            f.write("{0:<10f}	{1:40}".format(param[1], "Stall angle (deg)"))
            f.write(
                "{0:<10f}	{1:40}".format(
                    param[2], "Angle of attack for zero Cn for linear Cn curve (deg)"
                )
            )
            f.write(
                "{0:<10f}	{1:40}".format(
                    param[3],
                    "Cn slope for zero lift for linear Cn curve (1/rad)"
                )
            )
            f.write(
                "{0:<10f}	{1:40}".format(
                    param[4],
                    "Cn at stall value for positive angle of attack for linear Cn curve",
                )
            )
            f.write(
                "{0:<10f}	{1:40}".format(
                    param[5],
                    "Cn at stall value for negative angle of attack for linear Cn curve",
                )
            )
            f.write(
                "{0:<10f}	{1:40}".format(
                    param[6], "Angle of attack for minimum CD (deg)"
                )
            )
            f.write("{0:<10f}	{1:40}".format(param[7], "Minimum CD value"))
            for a, cl, cd, cm in zip(p.alpha, p.cl, p.cd, p.cm):
                f.write("{:<10f}	{:<10f}	{:<10f}	{:<10f}".format(a, cl, cd, cm))
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
