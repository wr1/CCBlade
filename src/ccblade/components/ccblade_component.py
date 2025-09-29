#!/usr/bin/env python
# encoding: utf-8
"""
ccblade_component.py

Created by Andrew Ning on 2/21/2012.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np
from openmdao.api import ExplicitComponent
from scipy.interpolate import PchipInterpolator

from ..airfoil.airfoil import CCAirfoil
from ..airfoil.airfoil import Airfoil
from ..utils.csystem import DirectionVector

cosd = lambda x: np.cos(np.deg2rad(x))
sind = lambda x: np.sin(np.deg2rad(x))


class CCBladeGeometry(ExplicitComponent):
    """Compute some geometric properties of the turbine based on the tip radius,
    precurve, presweep, and precone.

    Parameters
    ----------
    Rtip : float
        Rotor tip radius.
    precurve_in : numpy array[n_span]
        Prebend distribution along the span.
    presweep_in : numpy array[n_span]
        Presweep distribution along the span.
    precone : float
        Precone angle.

    Returns
    -------
    R : float
        Rotor radius.
    diameter : float
        Rotor diameter.
    precurveTip : float
        Precurve value at the rotor tip.
    presweepTip : float
        Presweep value at the rotor tip.
    """

    def initialize(self):
        self.options.declare("n_span")

    def setup(self):
        n_span = self.options["n_span"]

        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("precurve_in", val=np.zeros(n_span), units="m")
        self.add_input("presweep_in", val=np.zeros(n_span), units="m")
        self.add_input("precone", val=0.0, units="deg")

        self.add_output("R", val=0.0, units="m")
        self.add_output("diameter", val=0.0, units="m")
        self.add_output("precurveTip", val=0.0, units="m")
        self.add_output("presweepTip", val=0.0, units="m")

        self.declare_partials("R", ["Rtip", "precone"])
        self.declare_partials("diameter", ["Rtip", "precone"])

        self.declare_partials(
            ["R", "diameter"], "precurve_in", rows=[0], cols=[n_span - 1]
        )

        self.declare_partials(
            "precurveTip", "precurve_in", val=1.0, rows=[0], cols=[n_span - 1]
        )
        self.declare_partials(
            "presweepTip", "presweep_in", val=1.0, rows=[0], cols=[n_span - 1]
        )

    def compute(self, inputs, outputs):
        Rtip = inputs["Rtip"]
        precone = inputs["precone"]

        outputs["precurveTip"] = inputs["precurve_in"][-1]
        outputs["presweepTip"] = inputs["presweep_in"][-1]

        outputs["R"] = Rtip * cosd(precone) + outputs["precurveTip"] * sind(precone)
        outputs["diameter"] = outputs["R"] * 2

    def compute_partials(self, inputs, J):
        Rtip = inputs["Rtip"]
        precone = inputs["precone"]
        precurveTip = inputs["precurve_in"][-1]

        J["R", "precurve_in"] = sind(precone)
        J["R", "Rtip"] = cosd(precone)
        J["R", "precone"] = (
            (-Rtip * sind(precone) + precurveTip * cosd(precone)) * np.pi / 180.0
        )

        J["diameter", "precurve_in"] = 2.0 * J["R", "precurve_in"]
        J["diameter", "Rtip"] = 2.0 * J["R", "Rtip"]
        J["diameter", "precone"] = 2.0 * J["R", "precone"]


class CCBladeLoads(ExplicitComponent):
    """Compute the aerodynamic forces along the blade span given a rotor speed,
    pitch angle, and wind speed.

    This component instantiates and calls a CCBlade instance to compute the loads.
    Analytic derivatives are provided for all inptus except all airfoils*,
    mu, rho, and shearExp.

    Parameters
    ----------
    V_load : float
        Hub height wind speed.
    Omega_load : float
        Rotor rotation speed.
    pitch_load : float
        Blade pitch setting.
    azimuth_load : float
        Blade azimuthal location.
    r : numpy array[n_span]
        Radial locations where blade is defined. Should be increasing and not
        go all the way to hub or tip.
    chord : numpy array[n_span]
        Chord length at each section.
    theta : numpy array[n_span]
        Twist angle at each section (positive decreases angle of attack).
    Rhub : float
        Hub radius.
    Rtip : float
        Tip radius.
    hub_height : float
        Hub height.
    precone : float
        Precone angle.
    tilt : float
        Shaft tilt.
    yaw : float
        Yaw error.
    precurve : numpy array[n_span]
        Precurve at each section.
    precurveTip : float
        Precurve at tip.
    airfoils_cl : numpy array[n_span, n_aoa, n_Re, n_tab]
        Lift coefficients, spanwise.
    airfoils_cd : numpy array[n_span, n_aoa, n_Re, n_tab]
        Drag coefficients, spanwise.
    airfoils_cm : numpy array[n_span, n_aoa, n_Re, n_tab]
        Moment coefficients, spanwise.
    airfoils_aoa : numpy array[n_aoa]
        Angle of attack grid for polars.
    airfoils_Re : numpy array[n_Re]
        Reynolds numbers of polars.
    nBlades : int
        Number of blades
    rho : float
        Density of air
    mu : float
        Dynamic viscosity of air
    shearExp : float
        Shear exponent.
    nSector : int
        Number of sectors to divide rotor face into in computing thrust and power.
    tiploss : boolean
        Include Prandtl tip loss model.
    hubloss : boolean
        Include Prandtl hub loss model.
    wakerotation : boolean
        Include effect of wake rotation (i.e., tangential induction factor is nonzero).
    usecd : boolean
        Use drag coefficient in computing induction factors.

    Returns
    -------
    loads_r : numpy array[n_span]
         Radial positions along blade going toward tip.
    loads_Px : numpy array[n_span]
         Distributed loads in blade-aligned x-direction.
    loads_Py : numpy array[n_span]
         Distributed loads in blade-aligned y-direction.
    loads_Pz : numpy array[n_span]
         Distributed loads in blade-aligned z-direction.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds
        self.n_tab = n_tab = rotorse_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1

        # inputs
        self.add_input("V_load", val=20.0, units="m/s")
        self.add_input("Omega_load", val=0.0, units="rpm")
        self.add_input("pitch_load", val=0.0, units="deg")
        self.add_input("azimuth_load", val=0.0, units="deg")

        self.add_input("r", val=np.zeros(n_span), units="m")
        self.add_input("chord", val=np.zeros(n_span), units="m")
        self.add_input("theta", val=np.zeros(n_span), units="deg")
        self.add_input("Rhub", val=0.0, units="m")
        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("precone", val=0.0, units="deg")
        self.add_input("tilt", val=0.0, units="deg")
        self.add_input("yaw", val=0.0, units="deg")
        self.add_input("precurve", val=np.zeros(n_span), units="m")
        self.add_input("precurveTip", val=0.0, units="m")

        # parameters
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)))

        self.add_discrete_input("nBlades", val=0)
        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=0.0, units="kg/(m*s)")
        self.add_input("shearExp", val=0.0)
        self.add_discrete_input("nSector", val=4)
        self.add_discrete_input("tiploss", val=True)
        self.add_discrete_input("hubloss", val=True)
        self.add_discrete_input("wakerotation", val=True)
        self.add_discrete_input("usecd", val=True)

        # outputs
        self.add_output("loads_r", val=np.zeros(n_span), units="m")
        self.add_output("loads_Px", val=np.zeros(n_span), units="N/m")
        self.add_output("loads_Py", val=np.zeros(n_span), units="N/m")
        self.add_output("loads_Pz", val=np.zeros(n_span), units="N/m")

        arange = np.arange(n_span)
        self.declare_partials("loads_r", "r", val=1.0, rows=arange, cols=arange)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]
        azimuth_load = inputs["azimuth_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
            )

        ccblade = CCBlade(
            r,
            chord,
            theta,
            af,
            Rhub,
            Rtip,
            B,
            rho,
            mu,
            precone,
            tilt,
            yaw,
            shearExp,
            hub_height,
            nSector,
            precurve,
            precurveTip,
            tiploss=tiploss,
            hubloss=hubloss,
            wakerotation=wakerotation,
            usecd=usecd,
        )

        # distributed loads
        loads = ccblade.distributedAeroLoads(
            V_load, Omega_load, pitch_load, azimuth_load
        )
        Np = loads["Np"]
        Tp = loads["Tp"]

        # unclear why we need this output at all
        outputs["loads_r"] = r

        # conform to blade-aligned coordinate system
        outputs["loads_Px"] = Np
        outputs["loads_Py"] = -Tp
        outputs["loads_Pz"][:] = 0.0


class CCBladeTwist(ExplicitComponent):
    """Compute twist distribution for given operational conditions."""

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]
        self.n_span = n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        # self.n_af          = n_af      = af_init_options['n_af'] # Number of airfoils
        self.n_aoa = n_aoa = modeling_options["WISDEM"]["RotorSE"][
            "n_aoa"
        ]  # Number of angle of attacks
        self.n_Re = n_Re = modeling_options["WISDEM"]["RotorSE"][
            "n_Re"
        ]  # Number of Reynolds, so far hard set at 1
        self.n_tab = n_tab = modeling_options["WISDEM"]["RotorSE"][
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
        n_opt_chord = opt_options["design_variables"]["blade"]["aero_shape"]["chord"][
            "n_opt"
        ]
        n_opt_twist = opt_options["design_variables"]["blade"]["aero_shape"]["twist"][
            "n_opt"
        ]

        # Inputs
        self.add_input("Uhub", val=9.0, units="m/s", desc="Undisturbed wind speed")

        self.add_input("tsr", val=0.0, desc="Tip speed ratio")
        self.add_input("pitch", val=0.0, units="deg", desc="Pitch angle")
        self.add_input(
            "r",
            val=np.zeros(n_span),
            units="m",
            desc="radial locations where blade is defined (should be increasing and not go all the way to hub or tip)",
        )
        self.add_input(
            "s_opt_chord",
            val=np.zeros(n_opt_chord),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord",
        )
        self.add_input(
            "s_opt_theta",
            val=np.zeros(n_opt_twist),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist",
        )
        self.add_input(
            "chord",
            val=np.zeros(n_span),
            units="m",
            desc="chord length at each section",
        )
        self.add_input(
            "theta_in",
            val=np.zeros(n_span),
            units="rad",
            desc="twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input(
            "aoa_op",
            val=np.pi * np.ones(n_span),
            desc="1D array with the operational angles of attack for the airfoils along blade span.",
            units="rad",
        )
        self.add_input(
            "airfoils_aoa",
            val=np.zeros((n_aoa)),
            units="deg",
            desc="angle of attack grid for polars",
        )
        self.add_input(
            "airfoils_cl",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="lift coefficients, spanwise",
        )
        self.add_input(
            "airfoils_cd",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="drag coefficients, spanwise",
        )
        self.add_input(
            "airfoils_cm",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="moment coefficients, spanwise",
        )
        self.add_input(
            "airfoils_Re", val=np.zeros((n_Re)), desc="Reynolds numbers of polars"
        )
        self.add_input("Rhub", val=0.0, units="m", desc="hub radius")
        self.add_input("Rtip", val=0.0, units="m", desc="tip radius")
        self.add_input(
            "rthick",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )
        self.add_input(
            "precurve", val=np.zeros(n_span), units="m", desc="precurve at each section"
        )
        self.add_input("precurveTip", val=0.0, units="m", desc="precurve at tip")
        self.add_input(
            "presweep", val=np.zeros(n_span), units="m", desc="presweep at each section"
        )
        self.add_input("presweepTip", val=0.0, units="m", desc="presweep at tip")
        self.add_input("hub_height", val=0.0, units="m", desc="hub height")
        self.add_input(
            "precone",
            val=0.0,
            units="deg",
            desc="precone angle",
        )
        self.add_input(
            "tilt",
            val=0.0,
            units="deg",
            desc="shaft tilt",
        )
        self.add_input(
            "yaw",
            val=0.0,
            units="deg",
            desc="yaw error",
        )
        self.add_discrete_input("nBlades", val=0, desc="number of blades")
        self.add_input("rho", val=1.225, units="kg/m**3", desc="density of air")
        self.add_input(
            "mu", val=1.81e-5, units="kg/(m*s)", desc="dynamic viscosity of air"
        )
        self.add_input("shearExp", val=0.0, desc="shear exponent")
        self.add_discrete_input(
            "nSector",
            val=4,
            desc="number of sectors to divide rotor face into in computing thrust and power",
        )
        self.add_discrete_input(
            "tiploss", val=True, desc="include Prandtl tip loss model"
        )
        self.add_discrete_input(
            "hubloss", val=True, desc="include Prandtl hub loss model"
        )
        self.add_discrete_input(
            "wakerotation",
            val=True,
            desc="include effect of wake rotation (i.e., tangential induction factor is nonzero)",
        )
        self.add_discrete_input(
            "usecd",
            val=True,
            desc="use drag coefficient in computing induction factors",
        )

        # Outputs
        self.add_output(
            "theta",
            val=np.zeros(n_span),
            units="rad",
            desc="Twist angle at each section (positive decreases angle of attack)",
        )
        self.add_output("CP", val=0.0, desc="Rotor power coefficient")
        self.add_output("CM", val=0.0, desc="Blade flapwise moment coefficient")

        self.add_output(
            "local_airfoil_velocities",
            val=np.zeros(n_span),
            desc="Local relative velocities for the airfoils",
            units="m/s",
        )

        self.add_output("P", val=0.0, units="W", desc="Rotor aerodynamic power")
        self.add_output("T", val=0.0, units="N*m", desc="Rotor aerodynamic thrust")
        self.add_output("Q", val=0.0, units="N*m", desc="Rotor aerodynamic torque")
        self.add_output("M", val=0.0, units="N*m", desc="Blade root flapwise moment")

        self.add_output(
            "a", val=np.zeros(n_span), desc="Axial induction  along blade span"
        )
        self.add_output(
            "ap", val=np.zeros(n_span), desc="Tangential induction along blade span"
        )
        self.add_output(
            "alpha",
            val=np.zeros(n_span),
            units="deg",
            desc="Angles of attack along blade span",
        )
        self.add_output(
            "cl", val=np.zeros(n_span), desc="Lift coefficients along blade span"
        )
        self.add_output(
            "cd", val=np.zeros(n_span), desc="Drag coefficients along blade span"
        )
        n_opt = opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]
        self.add_output(
            "cl_n_opt", val=np.zeros(n_opt), desc="Lift coefficients along blade span"
        )
        self.add_output(
            "cd_n_opt", val=np.zeros(n_opt), desc="Drag coefficients along blade span"
        )
        self.add_output(
            "Px_b",
            val=np.zeros(n_span),
            units="N/m",
            desc="Distributed loads in blade-aligned x-direction",
        )
        self.add_output(
            "Py_b",
            val=np.zeros(n_span),
            units="N/m",
            desc="Distributed loads in blade-aligned y-direction",
        )
        self.add_output(
            "Pz_b",
            val=np.zeros(n_span),
            units="N/m",
            desc="Distributed loads in blade-aligned z-direction",
        )
        self.add_output(
            "Px_af",
            val=np.zeros(n_span),
            units="N/m",
            desc="Distributed loads in airfoil x-direction",
        )
        self.add_output(
            "Py_af",
            val=np.zeros(n_span),
            units="N/m",
            desc="Distributed loads in airfoil y-direction",
        )
        self.add_output(
            "Pz_af",
            val=np.zeros(n_span),
            units="N/m",
            desc="Distributed loads in airfoil z-direction",
        )
        self.add_output(
            "LiftF", val=np.zeros(n_span), units="N/m", desc="Distributed lift force"
        )
        self.add_output(
            "DragF", val=np.zeros(n_span), units="N/m", desc="Distributed drag force"
        )
        self.add_output(
            "L_n_opt", val=np.zeros(n_opt), units="N/m", desc="Distributed lift force"
        )
        self.add_output(
            "D_n_opt", val=np.zeros(n_opt), units="N/m", desc="Distributed drag force"
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Create Airfoil class instances
        af = [None] * self.n_span
        for i in range(self.n_span):
            if self.n_tab > 1:
                ref_tab = int(np.floor(self.n_tab / 2))
                af[i] = CCAirfoil(
                    inputs["airfoils_aoa"],
                    inputs["airfoils_Re"],
                    inputs["airfoils_cl"][i, :, :, ref_tab],
                    inputs["airfoils_cd"][i, :, :, ref_tab],
                    inputs["airfoils_cm"][i, :, :, ref_tab],
                )
            else:
                af[i] = CCAirfoil(
                    inputs["airfoils_aoa"],
                    inputs["airfoils_Re"],
                    inputs["airfoils_cl"][i, :, :, 0],
                    inputs["airfoils_cd"][i, :, :, 0],
                    inputs["airfoils_cm"][i, :, :, 0],
                )

        # Create the CCBlade class instance
        ccblade = CCBlade(
            inputs["r"],
            inputs["chord"],
            np.zeros_like(inputs["chord"]),
            af,
            inputs["Rhub"],
            inputs["Rtip"],
            discrete_inputs["nBlades"],
            inputs["rho"],
            inputs["mu"],
            inputs["precone"],
            inputs["tilt"],
            inputs["yaw"],
            inputs["shearExp"],
            inputs["hub_height"],
            discrete_inputs["nSector"],
            inputs["precurve"],
            inputs["precurveTip"],
            inputs["presweep"],
            inputs["presweepTip"],
            discrete_inputs["tiploss"],
            discrete_inputs["hubloss"],
            discrete_inputs["wakerotation"],
            discrete_inputs["usecd"],
        )

        Omega = inputs["tsr"] * inputs["Uhub"] / inputs["r"][-1] * 30.0 / np.pi

        if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"][
            "twist"
        ]["inverse"]:
            if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"][
                "twist"
            ]["flag"]:
                raise Exception(
                    "Twist cannot be simultaneously optimized and set to be defined inverting the BEM equations. Please check your analysis options yaml."
                )
            # Find cl and cd along blade span. Mix inputs from the airfoil INN (if active) and the airfoil for max efficiency
            cl = np.zeros(self.n_span)
            cd = np.zeros(self.n_span)
            alpha = np.zeros(self.n_span)
            margin2stall = (
                self.options["opt_options"]["constraints"]["blade"]["stall"]["margin"]
                * 180.0
                / np.pi
            )
            Re = np.array(
                Omega * inputs["r"] * inputs["chord"] * inputs["rho"] / inputs["mu"]
            )
            aoa_op = inputs["aoa_op"]
            for i in range(self.n_span):
                # Use the required angle of attack if defined. If it isn't defined (==pi), then take the stall point minus the margin
                if abs(aoa_op[i] - np.pi) < 1.0e-4:
                    af[i].eval_unsteady(
                        inputs["airfoils_aoa"],
                        inputs["airfoils_cl"][i, :, 0, 0],
                        inputs["airfoils_cd"][i, :, 0, 0],
                        inputs["airfoils_cm"][i, :, 0, 0],
                    )
                    alpha[i] = (af[i].unsteady["alpha1"] - margin2stall) / 180.0 * np.pi
                else:
                    alpha[i] = aoa_op[i]
                cl[i], cd[i] = af[i].evaluate(alpha[i], Re[i])

            # Overwrite aoa of high thickness airfoils at blade root
            idx_min = [i for i, thk in enumerate(inputs["rthick"]) if thk < 95.0][0]
            alpha[0:idx_min] = alpha[idx_min]

            # Call ccblade in inverse mode for desired alpha, cl, and cd along blade span
            ccblade.inverse_analysis = True
            ccblade.alpha = alpha
            ccblade.cl = cl
            ccblade.cd = cd
            _ = ccblade.evaluate(
                [inputs["Uhub"]], [Omega], [inputs["pitch"]], coefficients=False
            )

            # Cap twist root region to 20 degrees
            for i in range(len(ccblade.theta)):
                if ccblade.theta[-i - 1] > 20.0 / 180.0 * np.pi:
                    ccblade.theta[0 : len(ccblade.theta) - i] = 20.0 / 180.0 * np.pi
                    break
        else:
            ccblade.theta = inputs["theta_in"]

        # Smooth out twist profile if we're doing inverse and inn_af design
        if (
            self.options["opt_options"]["design_variables"]["blade"]["aero_shape"][
                "twist"
            ]["inverse"]
            and self.options["modeling_options"]["WISDEM"]["RotorSE"]["inn_af"]
        ):
            n_opt = self.options["opt_options"]["design_variables"][
                "blade"
            ]["aero_shape"]["twist"]["n_opt"]
            training_theta = np.copy(ccblade.theta)
            s = (inputs["r"] - inputs["r"][0]) / (inputs["r"][-1] - inputs["r"][0])

            twist_spline = PchipInterpolator(s, training_theta)
            theta_opt = twist_spline(inputs["s_opt_theta"])

            twist_spline = PchipInterpolator(inputs["s_opt_theta"], theta_opt)
            theta_full = twist_spline(s)
            ccblade.theta = theta_full

        # Turn off the inverse analysis
        ccblade.inverse_analysis = False

        # Call ccblade at azimuth 0 deg
        loads = ccblade.distributedAeroLoads(
            inputs["Uhub"][0], Omega[0], inputs["pitch"][0], 0.0
        )

        # Call ccblade evaluate (averaging across azimuth)
        myout = ccblade.evaluate(
            [inputs["Uhub"]], [Omega], [inputs["pitch"]], coefficients=True
        )
        CP, CMb, W = [myout[key] for key in ["CP", "CMb", "W"]]

        # Return twist angle
        outputs["theta"] = ccblade.theta
        outputs["CP"] = CP[0]
        outputs["CM"] = CMb[0]
        outputs["local_airfoil_velocities"] = np.nan_to_num(W)
        outputs["a"] = loads["a"]
        outputs["ap"] = loads["ap"]
        outputs["alpha"] = loads["alpha"]
        outputs["cl"] = loads["Cl"]
        outputs["cd"] = loads["Cd"]
        s = (inputs["r"] - inputs["r"][0]) / (inputs["r"][-1] - inputs["r"][0])
        outputs["cl_n_opt"] = np.interp(inputs["s_opt_theta"], s, loads["Cl"])
        outputs["cd_n_opt"] = np.interp(inputs["s_opt_theta"], s, loads["Cd"])
        # Forces in the blade coordinate system, pag 21 of https://www.nrel.gov/docs/fy13osti/58819.pdf
        outputs["Px_b"] = loads["Np"]
        outputs["Py_b"] = -loads["Tp"]
        outputs["Pz_b"] = 0 * loads["Np"]
        # Forces in the airfoil coordinate system, pag 21 of https://www.nrel.gov/docs/fy13osti/58819.pdf
        P_b = DirectionVector(loads["Np"], -loads["Tp"], 0)
        P_af = P_b.bladeToAirfoil(ccblade.theta * 180.0 / np.pi)
        outputs["Px_af"] = P_af.x
        outputs["Py_af"] = P_af.y
        outputs["Pz_af"] = P_af.z
        # Lift and drag forces
        F = P_b.bladeToAirfoil(
            ccblade.theta * 180.0 / np.pi + loads["alpha"] + inputs["pitch"]
        )
        outputs["LiftF"] = F.x
        outputs["DragF"] = F.y
        outputs["L_n_opt"] = np.interp(inputs["s_opt_theta"], s, F.x)
        outputs["D_n_opt"] = np.interp(inputs["s_opt_theta"], s, F.y)


class CCBladeEvaluate(ExplicitComponent):
    """Standalone component for CCBlade that is only a light wrapper on CCBlade()
    to run the instance evaluate and compute aerodynamic hub forces and moments, blade
    root flapwise moment, and power. The coefficients are also computed.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_init_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_init_options["n_span"]
        self.n_aoa = n_aoa = rotorse_init_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_init_options["n_Re"]  # Number of Reynolds
        self.n_tab = n_tab = rotorse_init_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1

        # inputs
        self.add_input("V_load", val=20.0, units="m/s")
        self.add_input("Omega_load", val=9.0, units="rpm")
        self.add_input("pitch_load", val=0.0, units="deg")

        self.add_input("r", val=np.zeros(n_span), units="m")
        self.add_input("chord", val=np.zeros(n_span), units="m")
        self.add_input("theta", val=np.zeros(n_span), units="deg")
        self.add_input("Rhub", val=0.0, units="m")
        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("precone", val=0.0, units="deg")
        self.add_input("tilt", val=0.0, units="deg")
        self.add_input("yaw", val=0.0, units="deg")
        self.add_input("precurve", val=np.zeros(n_span), units="m")
        self.add_input("precurveTip", val=0.0, units="m")
        self.add_input("presweep", val=np.zeros(n_span), units="m")
        self.add_input("presweepTip", val=0.0, units="m")

        # parameters
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)))

        self.add_discrete_input("nBlades", val=0)
        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=0.0, units="kg/(m*s)")
        self.add_input("shearExp", val=0.0)
        self.add_discrete_input("nSector", val=4)
        self.add_discrete_input("tiploss", val=True)
        self.add_discrete_input("hubloss", val=True)
        self.add_discrete_input("wakerotation", val=True)
        self.add_discrete_input("usecd", val=True)

        # outputs
        self.add_output("P", val=0.0, units="W", desc="Rotor aerodynamic power")
        self.add_output(
            "Mb", val=0.0, units="N/m", desc="Aerodynamic blade root flapwise moment"
        )
        self.add_output(
            "Fhub",
            val=np.zeros(3),
            units="N",
            desc="Aerodynamic forces at hub center in the hub c.s.",
        )
        self.add_output(
            "Mhub",
            val=np.zeros(3),
            units="N*m",
            desc="Aerodynamic moments at hub center in the hub c.s.",
        )
        self.add_output("CP", val=0.0, desc="Rotor aerodynamic power coefficient")
        self.add_output(
            "CMb", val=0.0, desc="Aerodynamic blade root flapwise moment coefficient"
        )
        self.add_output(
            "CFhub",
            val=np.zeros(3),
            desc="Aerodynamic force coefficients at hub center in the hub c.s.",
        )
        self.add_output(
            "CMhub",
            val=np.zeros(3),
            desc="Aerodynamic moment coefficients at hub center in the hub c.s.",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"]
        presweep = inputs["presweep"]
        presweepTip = inputs["presweepTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
            )

        ccblade = CCBlade(
            r,
            chord,
            theta,
            af,
            Rhub,
            Rtip,
            B,
            rho,
            mu,
            precone,
            tilt,
            yaw,
            shearExp,
            hub_height,
            nSector,
            precurve,
            precurveTip,
            presweep,
            presweepTip,
            tiploss=tiploss,
            hubloss=hubloss,
            wakerotation=wakerotation,
            usecd=usecd,
        )

        loads = ccblade.evaluate(V_load, Omega_load, pitch_load, coefficients=True)
        outputs["P"] = loads["P"]
        outputs["Mb"] = loads["Mb"]
        outputs["CP"] = loads["CP"]
        outputs["CMb"] = loads["CMb"]
        outputs["Fhub"] = np.array([loads["T"], loads["Y"], loads["Z"]])
        outputs["Mhub"] = np.array([loads["Q"], loads["My"], loads["Mz"]])
        outputs["CFhub"] = np.array([loads["CT"], loads["CY"], loads["CZ"]])
        outputs["CMhub"] = np.array([loads["CQ"], loads["CMy"], loads["CMz"]])
