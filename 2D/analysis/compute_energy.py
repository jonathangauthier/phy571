import numpy as np


def get_energy(gS,psi):
    """Get the energy of psi, and give the result of the hamiltonian applied on psi"""
    #calculate laplacian product
    hbar = gS.pot.hbar
    m = gS.pot.m
    Lx_psi = np.zeros((gS.Jx,gS.Jy),dtype=complex)
    Lx_psi[:,1:-1] = (psi[:,0:-2]+psi[:,2:]-2*psi[:,1:-1])/(gS.dx**2)
    Ly_psi = np.zeros((gS.Jx,gS.Jy),dtype=complex)
    Ly_psi[1:-1,:] = (psi[0:-2,:]+psi[2:,:]-2*psi[1:-1,:])/(gS.dx**2)
    H_psi =-(hbar)**2/(2*m)*Lx_psi -(hbar)**2/(2*m)*Ly_psi + gS.pot.Veff(gS.X,gS.Y,psi)*psi
    E = np.real(np.sum(np.conjugate(psi)*H_psi)*gS.dx*gS.dy)
    return E,H_psi