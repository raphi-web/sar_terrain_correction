from __future__ import annotations
import numpy as np
from . import io


def conversion(t_az: float) -> dict:
    """
   selects nearest sar conversion parameter set for a given azimuth time
    """
    T, sets = io.conversions()  # these are conversion tables
    k = int(np.argmin(np.abs(T - t_az))) # find closest entry
    return sets[k]


def ground_to_slant(gr: float | np.ndarray, t_az: float) -> np.ndarray:
    """
    converts ground range to slant range
    """
    c = conversion(t_az) # conversion coefficients of the polynominal
    gr = np.asarray(gr, dtype=float) # ground range
    v = c["grsr"][-1] # last coefficient

    # start the horner polynomial evaluation instead of a0 + a1*gr + a2*gr**2 + a3*gr**3
    # use the faster method(((a_3gr+a_2)gr+a_1)gr+a_0)

    for a in reversed(c["grsr"][:-1]):
        v = v * gr + a
    sr = v # return the slant range
    return sr


def slant_to_ground(sr: float | np.ndarray, t_az: float) -> np.ndarray:
    """
    converts slant range to ground range
    gr = P(sr - sr_0)
    """
    c = conversion(t_az)  # load conversion parameters
    sr = np.asarray(sr, dtype=float)
    x = sr - c["sr0"]  # polynominal has sr0 as origin.  in-swath 0 <= x <= ~166 km
    v = c["srgr"][-1] # last coefficient in the horner polynomial
    for a in reversed(c["srgr"][:-1]): # evaluate polynominal
        v = v * x + a

    gr = v
    return gr
