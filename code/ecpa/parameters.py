"""
Build the `params` dict that is passed to ELV() and flash functions.

make_params() returns an explicit dict of all scalar constants from
ecpa.constants — no globals() inspection required.
"""
import numpy as np
from .constants import (
    R, Na, kb, e, eps0,
    Mw, Ms, Mc,
    Tc1, Pc1, Tc4, Pc4,
    b1, b2, b3, b4,
    c11, a01, a02, a03, c14, a04,
    Tref,
    Akij, Bkij, Ckij, ASij, BSij, CSij,
    epsW, bettaW, kappaW,
    Z2, Z3, Sg2, Sg3, Rb2, Rb3, Penelouxs, Peneloux_CO2,
    Uref1s, Talfa1s, alfa1s,
    Uref4s, Talfa4s, alfa4s,
    dip01, pol1, pol2, pol3, pol4,
    GAMMA1, THETA1, zww,
)


def make_params():
    """Return all eCPA scalar constants as a dict of np.float64 values."""
    raw = dict(
        R=R, Na=Na, kb=kb, e=e, eps0=eps0,
        Mw=Mw, Ms=Ms, Mc=Mc,
        Tc1=Tc1, Pc1=Pc1, Tc4=Tc4, Pc4=Pc4,
        b1=b1, b2=b2, b3=b3, b4=b4,
        c11=c11, a01=a01, a02=a02, a03=a03, c14=c14, a04=a04,
        Tref=Tref,
        Akij=Akij, Bkij=Bkij, Ckij=Ckij,
        ASij=ASij, BSij=BSij, CSij=CSij,
        epsW=epsW, bettaW=bettaW, kappaW=kappaW,
        Z2=Z2, Z3=Z3, Sg2=Sg2, Sg3=Sg3, Rb2=Rb2, Rb3=Rb3,
        Penelouxs=Penelouxs, Peneloux_CO2=Peneloux_CO2,
        Uref1s=Uref1s, Talfa1s=Talfa1s, alfa1s=alfa1s,
        Uref4s=Uref4s, Talfa4s=Talfa4s, alfa4s=alfa4s,
        dip01=dip01, pol1=pol1, pol2=pol2, pol3=pol3, pol4=pol4,
        GAMMA1=GAMMA1, THETA1=THETA1, zww=zww,
    )
    return {k: np.float64(v) for k, v in raw.items()}
