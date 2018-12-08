import numpy as np
import math

def hermite_poly(x,n):
    """Give the value of the n-th physicists' Hermite polynomial at x"""
    coef = [0]*(n+1)
    coef[-1]=1
    return np.polynomial.hermite.hermval(x,coef)

def harmonic_state(gS,n):
    """Give the energy and the function corresponding to the n-th state of the linear harmonic oscillator"""
    E_n = gS.hbar*gS.pot.w*(n+1/2)
    x = gS.x
    psi_n = (2**n*math.factorial(n))**(-0.5)*(gS.m*gS.pot.w/(np.pi*gS.hbar))**0.25*np.exp(-gS.m*gS.pot.w*x**2/(2*gS.hbar))*\
            hermite_poly((gS.m*gS.pot.w/gS.hbar)**0.5*x,n)
    return E_n, psi_n
