"""CCBlade CLI"""

import argparse
import os

import numpy as np

from ccblade import CCAirfoil, CCBlade


def main():
    parser = argparse.ArgumentParser(description="CCBlade CLI")
    parser.add_argument("-u", "--u", type=float, default=10.0, help="Wind speed Uinf")
    parser.add_argument("-o", "--omega", type=float, default=12.1, help="Rotor speed Omega in RPM")
    parser.add_argument("-p", "--pitch", type=float, default=0.0, help="Pitch angle in degrees")
    parser.add_argument("-f", "--file", type=str, help="Airfoil file path")

    args = parser.parse_args()

    # geometry
    Rhub = 1.5
    Rtip = 63.0
    r = np.array(
        [
            2.8667,
            5.6000,
            8.3333,
            11.7500,
            15.8500,
            19.9500,
            24.0500,
            28.1500,
            32.2500,
            36.3500,
            40.4500,
            44.5500,
            48.6500,
            52.7500,
            56.1667,
            58.9000,
            61.6333,
        ]
    )
    chord = np.array(
        [
            3.542,
            3.854,
            4.167,
            4.557,
            4.652,
            4.458,
            4.249,
            4.007,
            3.748,
            3.502,
            3.256,
            3.010,
            2.764,
            2.518,
            2.313,
            2.086,
            1.419,
        ]
    )
    theta = np.array(
        [
            13.308,
            13.308,
            13.308,
            13.308,
            11.480,
            10.162,
            9.011,
            7.795,
            6.544,
            5.361,
            4.188,
            3.125,
            2.319,
            1.526,
            0.863,
            0.370,
            0.106,
        ]
    )
    B = 3
    rho = 1.225
    mu = 1.81206e-5

    if args.file:
        af = CCAirfoil.initFromAerodynFile(args.file)
        af = [af] * len(r)
    else:
        afinit = CCAirfoil.initFromAerodynFile
        basepath = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "5MW_AFFiles")
        airfoil_types = [0] * 8
        airfoil_types[0] = afinit(os.path.join(basepath, "Cylinder1.dat"))
        airfoil_types[1] = afinit(os.path.join(basepath, "Cylinder2.dat"))
        airfoil_types[2] = afinit(os.path.join(basepath, "DU40_A17.dat"))
        airfoil_types[3] = afinit(os.path.join(basepath, "DU35_A17.dat"))
        airfoil_types[4] = afinit(os.path.join(basepath, "DU30_A17.dat"))
        airfoil_types[5] = afinit(os.path.join(basepath, "DU25_A17.dat"))
        airfoil_types[6] = afinit(os.path.join(basepath, "DU21_A17.dat"))
        airfoil_types[7] = afinit(os.path.join(basepath, "NACA64_A17.dat"))
        af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
        af = [0] * len(r)
        for i in range(len(r)):
            af[i] = airfoil_types[af_idx[i]]

    rotor = CCBlade(
        r, chord, theta, af, Rhub, Rtip, B, rho, mu, precone=2.5, tilt=5.0, yaw=0.0, shearExp=0.2, hubHt=90.0
    )
    outputs = rotor.evaluate([args.u], [args.omega * np.pi / 30], [args.pitch])
    print(f"P: {outputs.P[0]} W")
    print(f"T: {outputs.T[0]} N")
    print(f"Q: {outputs.Q[0]} Nm")
