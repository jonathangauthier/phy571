import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt


src = "C:\\Users\\Maxime\\Documents\\phy571_project"

class groundState1DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 1D problem via Crank-Nicholson"""
    
    def __init__(self,xMin,xMax, tauMax, J, N, D, f, u0):
        """Equation: du/dt = D*d2u/dx2 + f(x,u)
                x in [xMin,xMax]
                t in [0,-1j*tauMax]
                J spatial points
                N temporal points
                u(x,0) = u0(x)
        """
        
        self.xMin = xMin
        self.xMax = xMax
        self.tauMax = tauMax
        self.J = J
        self.N = N
        self.D = D
        self.f = f
        self.u0 = u0
        
        self.x, self.dx = np.linspace(xMin,xMax, J, retstep=True)
        self.tau, self.dtau = np.linspace(0,tauMax, N, retstep=True)
        self.dt = -1j*self.dtau
        self.sigma = D*self.dt/(2*self.dx**2)
        
        A = np.zeros((3,J),dtype=complex)
        A[0,1] = 0
        A[0,2:] = -self.sigma
        A[1,:] = 1+2*self.sigma
        A[1,0] = 1
        A[1,-1] = 1
        A[2,:-2] = -self.sigma
        A[2,-2] = 0
        self.A = A
        
        diaUp = [self.sigma]*(self.J-1)
        diaUp[0] = 0
        diaDown = [self.sigma]*(self.J-1)
        diaDown[J-2] = 0
        dia = [1-2*self.sigma]*self.J
        dia[0] = 1
        dia[-1] = 1
        diagonals = [dia, diaUp, diaDown]
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
        C = np.dot(self.B,self.U)+self.dt*F
        C[0] = 0
        C[-1] = 0
        self.U = lin.solve_banded((1,1), self.A, C)
        
    def renorm(self):
        """Renormalize the current solution"""
        self.U /= (np.sum(np.abs(self.U)**2))**0.5
        
xMax = 15
tauMax = 10
J = 500
N = 100
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1
Ng = 1

def V(x):
    return w*x**2
    
def f(x,u):
    return 1/(1j*hbar)*(V(x)+ 0*Ng*np.abs(u)**2)*u

def u0(x):
    return np.exp(-x**2)
    
gS = groundState1DCN(-xMax,xMax,tauMax,J,N,D,f,u0)


def save_background():
    for k in range(50):
        gS.renorm()
        gS.step()
        
    np.savetxt(src+"bgFunc.txt",gS.U)


def evolution_to_background_static():
    for k in range(N):
        gS.renorm()
        plt.plot(gS.x, gS.U)
        gS.step()
        
    plt.show()


def evolution_to_background_anim():
    fig = plt.figure()
    line, = plt.plot([],[], label="$Wave \enspace function$")
    plt.xlim(-xMax,xMax)
    plt.ylim(-0.01,0.025)
    
    def make_frame(k):
        gS.renorm()
        line.set_data(gS.x, np.abs(gS.U)**2)
        #print(k)
        gS.step()
        return line,
        
        
    Vx = V(gS.x)
    Vx /= 1000
    plt.plot(gS.x, Vx, label="$Potential \enspace V$")
    
    ani = animation.FuncAnimation(fig, make_frame, interval = 1, blit=False)
    
    plt.title("$Evolution \enspace process \enspace to \enspace get \enspace the \enspace background \enspace state$")
    plt.legend()
    plt.show()


evolution_to_background_anim()

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