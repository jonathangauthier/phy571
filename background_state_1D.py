import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt

class groundState1DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 1D problem via Crank-Nicholson"""
    
    def __init__(self,xMax, tauMax, J, N, D, f, u0):
        """Equation: du/dt = D*d2u/dx2 + f(x,u)
                x in [0,xMax]
                t in [0,-1j*tauMax]
                J spatial points
                N temporal points
                u(x,0) = u0(x)
        """
            
        self.xMax = xMax
        self.tauMax = tauMax
        self.J = J
        self.N = N
        self.D = D
        self.f = f
        self.u0 = u0
        
        self.x, self.dx = np.linspace(0,xMax, J, retstep=True)
        self.tau, self.dtau = np.linspace(0,tauMax, N, retstep=True)
        self.dt = -1j*self.dtau
        self.sigma = D*self.dt/(2*self.dx**2)
        
        A = np.zeros((3,J),dtype=complex)
        A[0,1] = -2*self.sigma
        A[0,2:] = -self.sigma
        A[1,:] = 1+2*self.sigma
        A[2,:-1] = -self.sigma
        A[2,-1] = -2*self.sigma
        self.A = A
        
        diaUp = [self.sigma]*(self.J-1)
        diaUp[0] = 2*self.sigma
        diaDown = [self.sigma]*(self.J-1)
        diaDown[J-2] = 2*self.sigma
        diagonals = [[1-2*self.sigma]*self.J, diaUp, diaDown]
        self.B = np.array(diags(diagonals, [0, 1, -1]).toarray(),dtype=complex)
        
        self.U = np.array((np.vectorize(u0))(self.x),dtype=complex)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        self.isFirstStep = True
        
    def step(self):
        """Calculate the next step in the Crank-Nicholson algorithm"""
        if self.isFirstStep:
            F = (np.vectorize(self.f))(self.x, self.U)
            self.isFirstStep = False
        else:
            F = 3/2*(np.vectorize(self.f))(self.x, self.U)-1/2*(np.vectorize(self.f))(self.x, self.oldU)
        self.oldU = np.copy(self.U)
        self.U = lin.solve_banded((1,1), self.A, np.dot(self.B,self.U)+self.dt*F)
        
    def renorm(self):
        """Renormalize the current solution"""
        self.U /= (np.sum(np.abs(self.U)**2))**0.5
        
xMax = 10
tauMax = 10
J = 500
N = 100
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 30
Ng = 1

def V(x):
    return w*(x-xMax/2)**2
    
def f(x,u):
    return 1/(1j*hbar)*(V(x)*u+ Ng*np.abs(u)**2)*u

def u0(x):
    return np.exp(-(x-xMax/2)**2)
    
gS = groundState1DCN(xMax,tauMax,J,N,D,f,u0)


for k in range(N):
    gS.renorm()
    plt.plot(gS.x, gS.U)
    gS.step()
    
plt.show()


"""
gS.renorm()
M = np.zeros(N)
for k in range(N):
    M[k] = np.max(gS.U)
    gS.step()
    
    
def fit(x,a,b):
    return a*np.exp(-b*x)


popt,pcov = opt.curve_fit(fit,gS.tau[N//2:],M[N//2:])



plt.plot(gS.tau,M)
plt.plot(gS.tau, fit(gS.tau, *popt))

print("Ground state energy: ")
print(hbar*popt[1])

plt.show()
"""