import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt
import math
from  scipy.misc import derivative as der

src = "C:\\Users\\Maxime\\Documents\\phy571_project"


## Definition of the Algorithm

class groundState1DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 1D problem via Crank-Nicholson"""
    
    def __init__(self,xMin,xMax, tauMax, J, N, D, f, u0, a=-1j):
        """Equation: du/dt = D*d2u/dx2 + f(x,u)
                x in [xMin,xMax]
                t in [0,-1j*tauMax]
                J spatial points
                N temporal points
                u(x,0) = u0(x)
           Evolve in imaginary time for a=-1j and in real time for a=1
        """
        #definition of parameters
        self.xMin = xMin
        self.xMax = xMax
        self.tauMax = tauMax
        self.J = J
        self.N = N
        self.D = D
        self.f = f
        self.u0 = u0
        
        #definition of the grid
        self.x, self.dx = np.linspace(xMin,xMax, J, retstep=True)
        self.tau, self.dtau = np.linspace(0,tauMax, N, retstep=True)
        self.dt = a*self.dtau
        self.sigma = D*self.dt/(2*self.dx**2)
        
        #definition of the matrix A
        A = np.zeros((3,J),dtype=complex)
        A[0,1] = 0
        A[0,2:] = -self.sigma
        A[1,:] = 1+2*self.sigma
        A[1,0] = 1
        A[1,-1] = 1
        A[2,:-2] = -self.sigma
        A[2,-2] = 0
        self.A = A
        
        #definition of the matrix B
        diaUp = [self.sigma]*(self.J-1)
        diaUp[0] = 0
        diaDown = [self.sigma]*(self.J-1)
        diaDown[J-2] = 0
        dia = [1-2*self.sigma]*self.J
        dia[0] = 1
        dia[-1] = 1
        diagonals = [dia, diaUp, diaDown]
        self.B = np.array(diags(diagonals, [0, 1, -1]).toarray(),dtype=complex)
        
        #defintion of the initial condition
        self.U = np.array((np.vectorize(u0))(self.x),dtype=complex)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        self.isFirstStep = True
        
    def step(self):
        """Calculate the next step in the Crank-Nicholson algorithm"""
        #for the first step, we calculate only with the previous state
        if self.isFirstStep:
            F = (np.vectorize(self.f))(self.x, self.U)
            self.isFirstStep = False
        #else, we calculate with the previous step and the one before to keep the second order accuracy
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
        
     

## Definition of the problem

xMax = 30
tauMax = 10
J = 1000 #number of spatial points
N = 400 #number of temporal points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = 100*0.05/10
a=-1j #imaginary time


def V(x):
    """Potential of the BEC"""
    return 0.5*m*(w*x)**2
    
def Veff(x,u):
    """Effective potential of the BEC"""
    return V(x)+ Ng*np.abs(u)**2
    
def f(x,u):
    #return 1/(1j*hbar)*(V(x))*u
    return 1/(1j*hbar)*Veff(x,u)*u

def u0(x):
    """initial state"""
    return np.exp(-10*(x)**2)
    
gS = groundState1DCN(-xMax,xMax,tauMax,J,N,D,f,u0)


## Definition of the useful functions

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


def save_simulation():
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


def evolution_to_ground_anim(number_of_steps_per_frame, potential_size_factor=50):
    """Print the evolution of the BEC
            number_of_steps_per_frame: how many steps are calculated for each frame
            potential_size_factor: the potential is divided by this value to fit in the window
    """
    fig,ax = plt.subplots()
    line_wave_function, = plt.plot([],[], label="$Wave \enspace function$")
    plt.xlim(-xMax,xMax)
    plt.ylim(-0.01,0.5)
    
    Vx = V(gS.x)
    Vx /= potential_size_factor
    line_effective_potential, = plt.plot(gS.x, Vx, label="$Potential \enspace V$")
    
    line_effective_potential, = plt.plot([],[], label="$Effective \enspace potential \enspace Veff$")
    
    
    def make_frame(k):
        gS.renorm()
        line_wave_function.set_data(gS.x, np.abs(gS.U)**2)
        
        
        Veff_current = (np.vectorize(Veff))(gS.x, gS.U)
        Veff_current /= potential_size_factor
        line_effective_potential.set_data(gS.x, Veff_current)
        
        
        #print(k)
        #wave_function = np.abs(gS.U)**2
        #ax.set_title("{:1.1e}".format(max(wave_function)))
        
        for i in range(number_of_steps_per_frame):
            gS.renorm()
            gS.step()
        return line_wave_function,
        
        
    
    
    E_n, psi_n = harmonic_state(0)
    plt.plot(gS.x,np.abs(psi_n)**2, label="$Analytical \enspace solution \enspace level \enspace 0$")
    E_n, psi_n = harmonic_state(1)
    plt.plot(gS.x,np.abs(psi_n)**2, label="$Analytical \enspace solution \enspace level \enspace 1$")
    
    
    ani = animation.FuncAnimation(fig, make_frame, interval = 20, blit=False)
    
    plt.title("$Evolution \enspace process \enspace to \enspace get \enspace the \enspace background \enspace state$")
    plt.legend()
    plt.show()


def get_energy(psi):
    """Get the energy of a state, and give the result of the hamiltonian over the state"""
    #calculate laplacian product
    L_psi = np.zeros(J)
    L_psi[1:-1] = (psi[0:-2]+psi[2:]-2*psi[1:-1])/(gS.dx**2)
    H_psi =-(hbar)**2/(2*m)*L_psi + 1j*hbar*np.vectorize(f)(gS.x,psi)
    E = np.sum(np.conjugate(psi)*H_psi)*gS.dx
    return E,H_psi


## Evolution

#Animation of the evolution
evolution_to_ground_anim(10,10)


#Check with the ground state of the harmonic oscillator
"""
n = 0
E_theo, psi = harmonic_state(n)
E_calc,H_psi = get_energy(psi)

plt.plot(gS.x,np.abs(psi)**2,label="harmonic eigen state")
plt.plot(gS.x,[E_theo]*J,label="theoretical energy")
plt.plot(gS.x,[E_calc]*J,label="calculated energy")
plt.legend()
plt.show()
"""
