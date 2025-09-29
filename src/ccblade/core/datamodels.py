
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
