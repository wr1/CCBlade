
from pydantic import BaseModel


class BladeLoads(BaseModel):
    """Data model for blade loads."""
    Np: list[float]
    Tp: list[float]
    a: list[float]
    ap: list[float]
    alpha: list[float]
    Cl: list[float]
    Cd: list[float]
    Cn: list[float]
    Ct: list[float]
    W: list[float]
    Re: list[float]


class RotorGeometry(BaseModel):
    """Data model for rotor geometry."""
    r: list[float]
    chord: list[float]
    theta: list[float]
    Rhub: float
    Rtip: float
    B: int
    rho: float
    mu: float
    precone: float
    tilt: float
    yaw: float
    shearExp: float
    hubHt: float
    nSector: int
    precurve: list[float] = None
    precurveTip: float = 0.0
    presweep: list[float] = None
    presweepTip: float = 0.0
    tiploss: bool = True
    hubloss: bool = True
    wakerotation: bool = True
    usecd: bool = True
    iterRe: int = 1


class RotorOutputs(BaseModel):
    """Data model for rotor outputs."""
    P: list[float]
    T: list[float]
    Y: list[float]
    Z: list[float]
    Q: list[float]
    My: list[float]
    Mz: list[float]
    Mb: list[float]
    W: list[float]
    CP: list[float] = None
    CT: list[float] = None
    CY: list[float] = None
    CZ: list[float] = None
    CQ: list[float] = None
    CMy: list[float] = None
    CMz: list[float] = None
    CMb: list[float] = None
