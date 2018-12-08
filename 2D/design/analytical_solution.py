import numpy as np
import math

def hermite_poly(x,n):
    """Give the value of the n-th physicists' Hermite polynomial at x"""
    coef = [0]*(n+1)
    coef[-1]=1
    return np.polynomial.hermite.hermval(x,coef)


def harmonic_state_2D(gS,nx,ny):
    """Give the energy and the function corresponding of the linear harmonic oscillator in the nx-th state in x and in ny-th state in y"""
    E_nx = hbar*gS.pot.wx*(nx+1/2)
    E_ny = hbar*gS.pot.wy*(ny+1/2)
    E = E_nx + E_ny
    psi = (2**nx*math.factorial(nx))**(-0.5)*(m*gS.pot.wx/(np.pi*hbar))**0.25*np.exp(-m*gS.pot.wx*gS.X**2/(2*hbar))*\
               hermite_poly((m*gS.pot.wx/hbar)**0.5*gS.X,nx) *\
               (2**ny*math.factorial(ny))**(-0.5)*(m*gS.pot.wy/(np.pi*hbar))**0.25*np.exp(-m*gS.pot.wy*gS.Y**2/(2*hbar))*\
               hermite_poly((m*gS.pot.wy/hbar)**0.5*gS.Y,ny)
    return E, psi