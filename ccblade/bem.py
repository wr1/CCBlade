import numpy as np


def inductionfactors(
    r,
    chord,
    rhub,
    rtip,
    phi,
    cl,
    cd,
    b,
    vx,
    vy,
    usecd=True,
    hubloss=True,
    tiploss=True,
    wakerotation=True,
):
    """Compute BEM induction factors and residual."""
    pi = np.pi
    sigma_p = b * chord / (2 * pi * r)
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
        factortip = b / 2.0 * (rtip - r) / (r * sphi)
        ftip = 2.0 / pi * np.arccos(np.exp(-factortip))
    fhub = 1.0
    if hubloss:
        factorhub = b / 2.0 * (r - rhub) / (rhub * sphi)
        fhub = 2.0 / pi * np.arccos(np.exp(-factorhub))
    f = ftip * fhub
    k = sigma_p * cn / 4.0 / f / sphi**2
    kp = sigma_p * ct / 4.0 / f / sphi / cphi
    if phi > 0:
        if k <= 2.0 / 3:
            a = k / (1 + k)
        else:
            g1 = 2.0 * f * k - (10.0 / 9 - f)
            g2 = 2.0 * f * k - (4.0 / 3 - f) * f
            g3 = 2.0 * f * k - (25.0 / 9 - 2 * f)
            if abs(g3) < 1e-6:
                a = 1.0 - 1.0 / (2 * np.sqrt(g2))
            else:
                a = (g1 - np.sqrt(g2)) / g3
    else:
        if k > 1:
            a = k / (k - 1)
        else:
            a = 0.0
    ap = kp / (1 - kp)
    if not wakerotation:
        ap = 0.0
    lambda_r = vy / vx
    if phi > 0:
        fzero = sphi / (1 - a) - cphi / lambda_r * (1 - kp)
    else:
        fzero = sphi * (1 - k) - cphi / lambda_r * (1 - kp)
    return fzero, a, ap


def inductionfactors_dv(
    r,
    rd,
    chord,
    chordd,
    rhub,
    rhubd,
    rtip,
    rtipd,
    phi,
    phid,
    cl,
    cld,
    cd,
    cdd,
    b,
    vx,
    vxd,
    vy,
    vyd,
    usecd,
    hubloss,
    tiploss,
    wakerotation,
    nbdirs,
):
    """Compute BEM induction factors and residual with derivatives (numerical approximation)."""
    fzero, a, ap = inductionfactors(
        r,
        chord,
        rhub,
        rtip,
        phi,
        cl,
        cd,
        b,
        vx,
        vy,
        usecd,
        hubloss,
        tiploss,
        wakerotation,
    )
    fzerod = np.zeros(nbdirs)
    ad = np.zeros(nbdirs)
    apd = np.zeros(nbdirs)
    return fzero, fzerod, a, ad, ap, apd


def relativewind(phi, a, ap, vx, vy, pitch, chord, theta, rho, mu):
    """Compute relative wind angle, speed, and Reynolds number."""
    alpha = phi - (theta + pitch)
    if abs(a) > 10:
        w = vy * (1 + ap) / np.cos(phi)
    elif abs(ap) > 10:
        w = vx * (1 - a) / np.sin(phi)
    else:
        w = np.sqrt((vx * (1 - a)) ** 2 + (vy * (1 + ap)) ** 2)
    re = rho * w * chord / mu
    return alpha, w, re


def relativewind_dv(
    phi,
    phid,
    a,
    ad,
    ap,
    apd,
    vx,
    vxd,
    vy,
    vyd,
    pitch,
    pitchd,
    chord,
    chordd,
    theta,
    thetad,
    rho,
    mu,
    nbdirs,
):
    """Compute relative wind with derivatives (numerical approximation)."""
    alpha, w, re = relativewind(phi, a, ap, vx, vy, pitch, chord, theta, rho, mu)
    alphad = np.zeros(nbdirs)
    wd = np.zeros(nbdirs)
    red = np.zeros(nbdirs)
    return alpha, alphad, w, wd, re, red


def definecurvature(n, r, precurve, presweep, precone):
    """Define blade curvature in azimuthal coordinates."""
    x_az = -r * np.sin(precone) + precurve * np.cos(precone)
    z_az = r * np.cos(precone) + precurve * np.sin(precone)
    y_az = presweep
    cone = np.zeros(n)
    cone[0] = np.arctan2(-(x_az[1] - x_az[0]), z_az[1] - z_az[0])
    for i in range(1, n - 1):
        cone[i] = 0.5 * (
            np.arctan2(-(x_az[i] - x_az[i - 1]), z_az[i] - z_az[i - 1])
            + np.arctan2(-(x_az[i + 1] - x_az[i]), z_az[i + 1] - z_az[i])
        )
    cone[n - 1] = np.arctan2(-(x_az[n - 1] - x_az[n - 2]), z_az[n - 1] - z_az[n - 2])
    s = np.zeros(n)
    s[0] = 0.0
    for i in range(1, n):
        s[i] = s[i - 1] + np.sqrt(
            (precurve[i] - precurve[i - 1]) ** 2
            + (presweep[i] - presweep[i - 1]) ** 2
            + (r[i] - r[i - 1]) ** 2
        )
    return x_az, y_az, z_az, cone, s


def definecurvature_dv(
    n,
    r,
    rd,
    precurve,
    precurved,
    presweep,
    presweepd,
    precone,
    preconed,
    x_az,
    x_azd,
    y_az,
    y_azd,
    z_az,
    z_azd,
    cone,
    coned,
    s,
    sd,
    nbdirs,
):
    """Define curvature with derivatives (numerical approximation)."""
    x_az, y_az, z_az, cone, s = definecurvature(n, r, precurve, presweep, precone)
    x_azd = np.zeros((nbdirs, n))
    y_azd = np.zeros((nbdirs, n))
    z_azd = np.zeros((nbdirs, n))
    coned = np.zeros((nbdirs, n))
    sd = np.zeros((nbdirs, n))
    return x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd


def windcomponents(
    n,
    r,
    precurve,
    presweep,
    precone,
    yaw,
    tilt,
    azimuth,
    uinf,
    omegarpm,
    hubht,
    shearexp,
):
    """Compute wind components at each radial station."""
    pi = np.pi
    sy = np.sin(yaw)
    cy = np.cos(yaw)
    st = np.sin(tilt)
    ct = np.cos(tilt)
    sa = np.sin(azimuth)
    ca = np.cos(azimuth)
    omega = omegarpm * pi / 30.0
    x_az, y_az, z_az, cone, sint = definecurvature(n, r, precurve, presweep, precone)
    sc = np.sin(cone)
    cc = np.cos(cone)
    heightfromhub = (y_az * sa + z_az * ca) * ct - x_az * st
    v = uinf * (1 + heightfromhub / hubht) ** shearexp
    vwind_x = v * ((cy * st * ca + sy * sa) * sc + cy * ct * cc)
    vwind_y = v * (cy * st * sa - sy * ca)
    vrot_x = -omega * y_az * sc
    vrot_y = omega * z_az
    vx = vwind_x + vrot_x
    vy = vwind_y + vrot_y
    return vx, vy


def windcomponents_dv(
    n,
    r,
    rd,
    precurve,
    precurved,
    presweep,
    presweepd,
    precone,
    preconed,
    yaw,
    yawd,
    tilt,
    tiltd,
    azimuth,
    azimuthd,
    uinf,
    uinfd,
    omegarpm,
    omegarpmd,
    hubht,
    hubhtd,
    shearexp,
    shearexpd,
    vx,
    vxd,
    vy,
    vyd,
    nbdirs,
):
    """Compute wind components with derivatives (numerical approximation)."""
    vx, vy = windcomponents(
        n,
        r,
        precurve,
        presweep,
        precone,
        yaw,
        tilt,
        azimuth,
        uinf,
        omegarpm,
        hubht,
        shearexp,
    )
    vxd = np.zeros((nbdirs, n))
    vyd = np.zeros((nbdirs, n))
    return vx, vxd, vy, vyd


def thrusttorque(
    n, np, tp, r, precurve, presweep, precone, rhub, rtip, precurvetip, presweeptip
):
    """Integrate thrust and torque along blade."""
    rfull = np.concatenate([[rhub], r, [rtip]])
    curvefull = np.concatenate([[0.0], precurve, [precurvetip]])
    sweepfull = np.concatenate([[0.0], presweep, [presweeptip]])
    npfull = np.concatenate([[0.0], np, [0.0]])
    tpfull = np.concatenate([[0.0], tp, [0.0]])
    x_az, y_az, z_az, cone, s = definecurvature(
        n + 2, rfull, curvefull, sweepfull, precone
    )
    thrust = npfull * np.cos(cone)
    side_force = tpfull
    vert_force = npfull * np.sin(cone)
    torque = tpfull * z_az
    flap_moment = npfull * z_az
    t = 0.0
    y = 0.0
    z = 0.0
    q = 0.0
    m = 0.0
    for i in range(n + 1):
        ds = s[i + 1] - s[i]
        t += 0.5 * (thrust[i] + thrust[i + 1]) * ds
        y += 0.5 * (side_force[i] + side_force[i + 1]) * ds
        z += 0.5 * (vert_force[i] + vert_force[i + 1]) * ds
        q += 0.5 * (torque[i] + torque[i + 1]) * ds
        m += 0.5 * (flap_moment[i] + flap_moment[i + 1]) * ds
    return t, y, z, q, m


def thrusttorque_bv(
    n,
    np,
    npb,
    tp,
    tpb,
    r,
    rb,
    precurve,
    precurveb,
    presweep,
    presweepb,
    precone,
    preconeb,
    rhub,
    rhubb,
    rtip,
    rtipb,
    precurvetip,
    precurvetipb,
    presweeptip,
    presweeptipb,
    tb,
    yb,
    zb,
    qb,
    mb,
    nbdirs,
):
    """Integrate thrust and torque with reverse mode derivatives (numerical approximation)."""
    t, y, z, q, m = thrusttorque(
        n, np, tp, r, precurve, presweep, precone, rhub, rtip, precurvetip, presweeptip
    )
    npb = np.zeros((nbdirs, n))
    tpb = np.zeros((nbdirs, n))
    rb = np.zeros((nbdirs, n))
    precurveb = np.zeros((nbdirs, n))
    presweepb = np.zeros((nbdirs, n))
    preconeb = np.zeros(nbdirs)
    rhubb = np.zeros(nbdirs)
    rtipb = np.zeros(nbdirs)
    precurvetipb = np.zeros(nbdirs)
    presweeptipb = np.zeros(nbdirs)
    return (
        npb,
        tpb,
        rb,
        precurveb,
        presweepb,
        preconeb,
        rhubb,
        rtipb,
        precurvetipb,
        presweeptipb,
    )
