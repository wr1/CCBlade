from pydantic import BaseModel
from typing import List


class BladeLoads(BaseModel):
    """Data model for blade loads."""
    Np: List[float]
    Tp: List[float]
    a: List[float]
    ap: List[float]
    alpha: List[float]
    Cl: List[float]
    Cd: List[float]
    Cn: List[float]
    Ct: List[float]
    W: List[float]
    Re: List[float]


class RotorOutputs(BaseModel):
    """Data model for rotor outputs."""
    P: List[float]
    T: List[float]
    Y: List[float]
    Z: List[float]
    Q: List[float]
    My: List[float]
    Mz: List[float]
    Mb: List[float]
    W: List[float]
    CP: List[float] = None
    CT: List[float] = None
    CY: List[float] = None
    CZ: List[float] = None
    CQ: List[float] = None
    CMy: List[float] = None
    CMz: List[float] = None
    CMb: List[float] = None
