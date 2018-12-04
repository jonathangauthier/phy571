import numpy as np
import scipy.linalg as lin
from scipy.sparse import diags

## Definition of the Algorithm

class groundState2DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 2D problem via Crank-Nicholson"""
    
    def __init__(self,xMin,xMax, yMin,yMax, dtau, Jx, Jy, D, pot, a=-1j):
        """Equation: du/dt = D*(d2u/dx2+d2/dy2) + pot.f(x,u)
                x in [xMin,xMax]
                y in [yMin, yMax]
                dtau: temporal step (dt = a*dtau)
                Jx spatial points in x
                Jy spatial points in y
           Evolve in imaginary time for a=-1j and in real time for a=1
        """
        #definition of parameters
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.dtau = dtau
        self.Jx = Jx
        self.Jy = Jy
        self.D = D
        self.pot = pot
        self.a = a
        
        #definition of the grid
        self.x, self.dx = np.linspace(xMin,xMax, Jx, retstep=True)
        self.y, self.dy = np.linspace(yMin,yMax, Jy, retstep=True)
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.dt = a*self.dtau
        self.sigma_x = D*self.dt/(2*self.dx**2)
        self.sigma_y = D*self.dt/(2*self.dy**2)
        
        #definition of the matrix Ax
        self.Ax = create_A_matrix(self.Jx,self.sigma_x)
        
        #definition of the matrix Ay
        self.Ay = create_A_matrix(self.Jy,self.sigma_y)
        
        #definition of the matrix Bx
        self.Bx = create_B_matrix(self.Jx,self.sigma_x)
        
        #definition of the matrix By
        self.By = create_B_matrix(self.Jy,self.sigma_y)
        
        #definition of the initial condition
        self.U = np.zeros_like(self.X)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        self.isFirstStep = True
        
    def initialise_function(self,u0):
        self.U = np.array((np.vectorize(u0))(self.X,self.Y),dtype=complex)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        
    def initialise_grid(self,U0):
        self.U = np.copy(U0)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        
    def step(self):
        """Calculate the next step in the Crank-Nicholson algorithm"""
        #for the first step, we calculate only with the previous state
        if self.isFirstStep:
            F = (self.pot.f)(self.X, self.Y, self.U)
            self.isFirstStep = False
        #else, we calculate with the previous step and the one before to keep the second order accuracy
        else:
            F = 3/2*(self.pot.f)(self.X, self.Y, self.U)-1/2*(self.pot.f)(self.X, self.Y, self.oldU)
        self.oldU = np.copy(self.U)
        #Propagation in x
        C = self.Bx.dot(self.U)+0.5*self.dt*F
        C[0,:] = 0
        C[-1,:] = 0
        self.U = lin.solve_banded((1,1), self.Ax, C, check_finite=False)
        #Propagation in y
        C = (self.By.dot(self.U.T))+0.5*self.dt*F
        C[0,:] = 0
        C[-1,:] = 0
        self.U = lin.solve_banded((1,1), self.Ay, C, check_finite=False).T
    
    def change_a(self, a):
        self.dt = a*self.dtau
        self.sigma_x = self.D*self.dt/(2*self.dx**2)
        self.sigma_y = self.D*self.dt/(2*self.dy**2)
        
        #redefinition of the matrix Ax
        self.Ax = create_A_matrix(self.Jx,self.sigma_x)
        
        #redefinition of the matrix Ay
        self.Ax = create_A_matrix(self.Jy,self.sigma_y)
        
        #redefinition of the matrix Bx
        self.Bx = create_B_matrix(self.Jx,self.sigma_x)
        
        #redefinition of the matrix By
        self.By = create_B_matrix(self.Jy,self.sigma_y)
    
    def renorm(self,vortex=False,xv=0,yv=0):
        """Renormalize the current solution"""
        self.U /= (np.sum(np.abs(self.U)**2)*self.dx*self.dy)**0.5
        if vortex:
            self.U = np.abs(self.U)*np.exp(1j*np.angle((self.X-xv)+1j*(self.Y-yv)))
 
 
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