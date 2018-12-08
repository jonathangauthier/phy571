import numpy as np
import scipy.linalg as lin
from scipy.sparse import diags


class groundState1DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 1D problem via Crank-Nicholson"""
    
    def __init__(self,xMin,xMax, dtau, J, D, pot, u0, a=-1j,hbar=1,m=1):
        """Equation: du/dt = D*d2u/dx2 + pot.f(x,u)
                x in [xMin,xMax]
                dtau: temporal step (dt = a*dtau)
                J spatial points
                u(x,0) = u0(x)
                m: particle mass
           Evolve in imaginary time for a=-1j and in real time for a=1
        """
        #definition of parameters
        self.xMin = xMin
        self.xMax = xMax
        self.dtau = dtau
        self.J = J
        self.D = D
        self.pot = pot
        self.u0 = u0
        self.a = a
        self.hbar = hbar
        self.m = m
        
        #definition of the grid
        self.x, self.dx = np.linspace(xMin,xMax, J, retstep=True)
        self.dt = a*self.dtau
        self.sigma = D*self.dt/(2*self.dx**2)
        
        #definition of the matrix A
        self.A = create_A_matrix(self.J,self.sigma)
        
        #definition of the matrix B
        self.B = create_B_matrix(self.J,self.sigma)
        
        #definition of the initial condition
        self.U = np.array((np.vectorize(u0))(self.x),dtype=complex)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        self.isFirstStep = True
    
    def change_a(self,a):
        self.a = a
        
        self.dt = a*self.dtau
        self.sigma = self.D*self.dt/(2*self.dx**2)
        
        #redefinition of the matrix A
        self.A = create_A_matrix(self.J,self.sigma)
        
        #redefinition of the matrix B
        self.B = create_B_matrix(self.J,self.sigma)
        
    def step(self):
        """Calculate the next step in the Crank-Nicholson algorithm"""
        #for the first step, we calculate only with the previous state
        if self.isFirstStep:
            F = (self.pot.f)(self.x, self.U)
            self.isFirstStep = False
        #else, we calculate with the previous step and the one before to keep the second order accuracy
        else:
            F = 3/2*(self.pot.f)(self.x, self.U)-1/2*(self.pot.f)(self.x, self.oldU)
        self.oldU = np.copy(self.U)
        C = self.B.dot(self.U)+self.dt*F
        C[0] = 0
        C[-1] = 0
        self.U = lin.solve_banded((1,1), self.A, C)
        
    def renorm(self):
        """Renormalize the current solution"""
        self.U /= (np.sum(np.abs(self.U)**2)*self.dx)**0.5
 
 
def create_A_matrix(J,sigma):
    """Create a tridiagonal matrix in the format used in the CN algorithm for the left hand side"""
    A = np.zeros((3,J),dtype=complex)
    A[0,1] = 0
    A[0,2:] = -sigma
    A[1,:] = 1+2*sigma
    A[1,0] = 1
    A[1,-1] = 1
    A[2,:-2] = -sigma
    A[2,-2] = 0
    return A

def create_B_matrix(J,sigma):
    """Create a tridiagonal matrix in the format used in the CN algorithm for the right hand side"""
    diaUp = sigma*np.ones(J-1,dtype=complex)
    diaUp[0] = 0
    diaDown = sigma*np.ones(J-1,dtype=complex)
    diaDown[J-2] = 0
    dia = (1-2*sigma)*np.ones(J,dtype=complex)
    dia[0] = 1
    dia[-1] = 1
    return diags([diaUp,dia,diaDown], [1,0,-1], (J, J), format='csr')