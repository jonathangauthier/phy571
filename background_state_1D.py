import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt
import math
from  scipy.misc import derivative as der
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
        self.U /= (np.sum(np.abs(self.U)**2)*self.dx)**0.5
        
xMax = 30
tauMax = 10
J = 1000
N = 400
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1
Ng = 1

def V(x):
    return 0.5*m*(w*x)**2
    
def f(x,u):
    return 1/(1j*hbar)*(V(x))*u
    return 1/(1j*hbar)*(V(x)+ Ng*np.abs(u)**2)*u

def u0(x):
    return np.exp(-10*(x)**2)
    
gS = groundState1DCN(-xMax,xMax,tauMax,J,N,D,f,u0)


def hermite_poly(x,n):
    """Give the value of the n-th physicists' Hermite polynomial at x"""
    coef = [0]*(n+1)
    coef[-1]=1
    return np.polynomial.hermite.hermval(x,coef)

def harmonic_state(n):
    """Give the energy and the function corresponding to the n-th state of the linear harmonic oscillator"""
    E_n = hbar*w*(n+1/2)
    x = gS.x
    psi_n = (2**n*math.factorial(n))**(-0.5)*(m*w/(np.pi*hbar))**0.25*np.exp(-m*w*x**2/(2*hbar))*\
            hermite_poly((m*w/hbar)**0.5*x,n)
    return E_n, psi_n


def save_ground():
    for k in range(50):
        gS.renorm()
        gS.step()
        
    np.savetxt(src+"bgFunc.txt",gS.U)


def evolution_to_ground_static():
    for k in range(N):
        gS.renorm()
        plt.plot(gS.x, gS.U)
        gS.step()
        
    plt.show()


def evolution_to_ground_anim():
    fig,ax = plt.subplots()
    line, = plt.plot([],[], label="$Wave \enspace function$")
    plt.xlim(-xMax,xMax)
    plt.ylim(-0.01,0.5)
    
    def make_frame(k):
        gS.renorm()
        line.set_data(gS.x, np.abs(gS.U)**2)
        #print(k)
        wave_function = np.abs(gS.U)**2
        #ax.set_title("{:1.1e}".format(max(wave_function)))
        gS.step()
        return line,
        
        
    Vx = V(gS.x)
    Vx /= 100
    plt.plot(gS.x, Vx, label="$Potential \enspace V$")
    
    E_n, psi_n = harmonic_state(0)
    plt.plot(gS.x,np.abs(psi_n)**2, label="$Analytical \enspace solution$")
    E_n, psi_n = harmonic_state(1)
    plt.plot(gS.x,np.abs(psi_n)**2, label="$Analytical \enspace solution$")
    
    
    ani = animation.FuncAnimation(fig, make_frame, interval = 1, blit=False)
    
    plt.title("$Evolution \enspace process \enspace to \enspace get \enspace the \enspace background \enspace state$")
    plt.legend()
    plt.show()


evolution_to_ground_anim()

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

def compute_derivative(f,dx):
    "compute the derivative of f, where f is an array of the values of the function at points separated with dx"
    f_prime = np.zeros_like(f)
    f_prime[0] = (f[1]-f[0])/dx
    for i in range(1,gS.J-1):
        f_prime[i] = (f[i+1]-f[i-1])/(2*dx)
    f_prime[gS.J-1] = (f[gS.J-1]-f[gS.J-2])/dx
    return f_prime

def get_energy(psi):
    #compute the second derivative
    psi_prime = compute_derivative(psi,gS.dx)
    psi_second = compute_derivative(psi_prime,gS.dx)
    H_psi =-(hbar)**2/(2*m)*psi_second+1j*hbar*np.vectorize(f)(gS.x,psi)
    return H_psi

"""
n = 0
   
E, psi = harmonic_state(n)

plt.plot(gS.x,psi)
plt.plot(gS.x,get_energy(psi))
plt.plot(gS.x,get_energy(psi)/psi)
plt.plot(gS.x,[hbar*w*(n+1/2)]*J)
plt.show()
"""