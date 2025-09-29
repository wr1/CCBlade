import numpy as np


def inductionfactors(
    r,
    chord,
    rhub,
    rtip,
    phi,
    cl,
    cd,
    n_blades,
    vx,
    vy,
    usecd=True,
    hubloss=True,
    tiploss=True,
    wakerotation=True,
):
    """Compute BEM induction factors."""
    pi = np.pi
    sigma_p = n_blades * chord / (2 * pi * r)
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    if usecd:
        cn = cl * cphi + cd * sphi
        ct = cl * sphi - cd * cphi
    else:
        cn = cl * cphi
        ct = cl * sphi
    ftip = 1.0
    if tiploss:
        factortip = n_blades / 2.0 * (rtip - r) / (r * sphi)
        ftip = 2.0 / pi * np.arccos(np.exp(-factortip))
    fhub = 1.0
    if hubloss:
        factorhub = n_blades / 2.0 * (r - rhub) / (rhub * sphi)
        fhub = 2.0 / pi * np.arccos(np.exp(-factorhub))
    f = ftip * fhub
    k = sigma_p * cn / 4.0 / f / sphi**2
    kp = sigma_p * ct / 4.0 / f / sphi / cphi
    if phi > 0:
        if k <= 2.0 / 3:
            axial_induction = k / (1 + k)
        else:
            g1 = 2.0 * f * k - (10.0 / 9 - f)
            g2 = 2.0 * f * k - (4.0 / 3 - f) * f
            g3 = 2.0 * f * k - (25.0 / 9 - 2 * f)
            if abs(g3) < 1e-6:
                axial_induction = 1.0 - 1.0 / (2 * np.sqrt(g2))
            else:
                axial_induction = (g1 - np.sqrt(g2)) / g3
    else:
        if k > 1:
            axial_induction = k / (k - 1)
        else:
            axial_induction = 0.0
    tangential_induction = kp / (1 - kp)
    if not wakerotation:
        tangential_induction = 0.0
    lambda_r = vy / vx
    if phi > 0:
        fzero = sphi / (1 - axial_induction) - cphi / lambda_r * (1 - kp)
    else:
        fzero = sphi * (1 - k) - cphi / lambda_r * (1 - kp)
    return fzero, axial_induction, tangential_induction


def relativewind(phi, axial_induction, tangential_induction, vx, vy, pitch, chord, theta, rho, mu):
    """Compute relative wind."""
    alpha = phi - (theta + pitch)
    if abs(axial_induction) > 10:
        w = vy * (1 + tangential_induction) / np.cos(phi)
    elif abs(tangential_induction) > 10:
        w = vx * (1 - axial_induction) / np.sin(phi)
    else:
        w = np.sqrt((vx * (1 - axial_induction)) ** 2 + (vy * (1 + tangential_induction)) ** 2)
    re = rho * w * chord / mu
    return alpha, w, re


def definecurvature(n_stations, r, precurve, presweep, precone):
    """Define blade curvature."""
    x_az = -r * np.sin(precone) + precurve * np.cos(precone)
    z_az = r * np.cos(precone) + precurve * np.sin(precone)
    y_az = presweep
    cone = np.zeros(n_stations)
    cone[0] = np.arctan2(-(x_az[1] - x_az[0]), z_az[1] - z_az[0])
    for i in range(1, n_stations - 1):
        cone[i] = 0.5 * (
            np.arctan2(-(x_az[i] - x_az[i - 1]), z_az[i] - z_az[i - 1])
            + np.arctan2(-(x_az[i + 1] - x_az[i]), z_az[i + 1] - z_az[i])
        )
    cone[n_stations - 1] = np.arctan2(-(x_az[n_stations - 1] - x_az[n_stations - 2]), z_az[n_stations - 1] - z_az[n_stations - 2])
    s = np.zeros(n_stations)
    s[0] = 0.0
    for i in range(1, n_stations):
        s[i] = s[i - 1] + np.sqrt(
            (precurve[i] - precurve[i - 1]) ** 2
            + (presweep[i] - presweep[i - 1]) ** 2
            + (r[i] - r[i - 1]) ** 2
        )
    return x_az, y_az, z_az, cone, s


def windcomponents(
    n_stations,
    r,
    precurve,
    presweep,
    precone,
    yaw,
    tilt,
    azimuth,
    wind_speed,
    omega_rpm,
    hub_height,
    shear_exp,
):
    """Compute wind components."""
    pi = np.pi
    sy = np.sin(yaw)
    cy = np.cos(yaw)
    st = np.sin(tilt)
    ct = np.cos(tilt)
    sa = np.sin(azimuth)
    ca = np.cos(azimuth)
    omega = omega_rpm * pi / 30.0
    x_az, y_az, z_az, cone, sint = definecurvature(n_stations, r, precurve, presweep, precone)
    sc = np.sin(cone)
    cc = np.cos(cone)
    heightfromhub = (y_az * sa + z_az * ca) * ct - x_az * st
    v = wind_speed * (1 + heightfromhub / hub_height) ** shear_exp
    vwind_x = v * ((cy * st * ca + sy * sa) * sc + cy * ct * cc)
    vwind_y = v * (cy * st * sa - sy * ca)
    vrot_x = -omega * y_az * sc
    vrot_y = omega * z_az
    vx = vwind_x + vrot_x
    vy = vwind_y + vrot_y
    return vx, vy


def thrusttorque(n_stations, normal_force, tangential_force, r, precurve, presweep, precone, rhub, rtip, precurvetip, presweeptip):
    """Integrate thrust and torque."""
    rfull = np.concatenate([[rhub], r, [rtip]])
    curvefull = np.concatenate([[0.0], precurve, [precurvetip]])
    sweepfull = np.concatenate([[0.0], presweep, [presweeptip]])
    Npfull = np.concatenate([[0.0], normal_force, [0.0]])
    Tpfull = np.concatenate([[0.0], tangential_force, [0.0]])
    x_az, y_az, z_az, cone, s = definecurvature(
        n_stations + 2, rfull, curvefull, sweepfull, precone
    )
    thrust = Npfull * np.cos(cone)
    side_force = Tpfull
    vert_force = Npfull * np.sin(cone)
    torque = Tpfull * z_az
    flap_moment = Npfull * z_az
    t = 0.0
    y = 0.0
    z = 0.0
    q = 0.0
    m = 0.0
    for i in range(n_stations + 1):
        ds = s[i + 1] - s[i]
        t += 0.5 * (thrust[i] + thrust[i + 1]) * ds
        y += 0.5 * (side_force[i] + side_force[i + 1]) * ds
        z += 0.5 * (vert_force[i] + vert_force[i + 1]) * ds
        q += 0.5 * (torque[i] + torque[i + 1]) * ds
        m += 0.5 * (flap_moment[i] + flap_moment[i + 1]) * ds
    return t, y, z, q, m
