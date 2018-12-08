import numpy as np

def get_energy(gS,psi):
    """Give the energy of a state, and give the result of the hamiltonian over the state"""
    #calculate laplacian product
    L_psi = np.zeros(gS.J,dtype=complex)
    L_psi[1:-1] = (psi[0:-2]+psi[2:]-2*psi[1:-1])/(gS.dx**2)
    H_psi =-(gS.hbar)**2/(2*gS.m)*L_psi + 1j*gS.hbar*gS.pot.f(gS.x,psi)
    E = np.real(np.sum(np.conjugate(psi)*H_psi)*gS.dx)
    return E,H_psi