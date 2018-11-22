import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

## Definition of the Algorithm

class Spectral:
    '''Spectral stepping method for solving PDEs'''
    def __init__(self, L, tauMax, J, N, D, f, u0, a=-1j):
        """Equation: du/dt = D*d2u/dx2 + f(x,u)
                x in [xMin,xMax]
                t in [0,-1j*tauMax]
                J spatial points
                N temporal points
                u(x,0) = u0(x)
           Evolve in imaginary time for a=-1j and in real time for a=1
        """
        
        # Definition of parameters
        self.L = L 
        self.tauMax = tauMax
        self.J = J
        self.N = N
        self.D = D
        self.f = f
        self.u0 = u0
        
        #definition of the grid
        self.x, self.dx = np.linspace(-self.L/2, self.L/2, J, retstep=True)
        self.tau, self.dtau = np.linspace(0,tauMax, N, retstep=True)
        self.dt = a*self.dtau
        self.sigma = D*self.dt/(2*self.dx**2)
        self.k = 2 * np.pi * np.fft.fftfreq(self.J)
        
        #defintion of the initial condition
        self.U = np.array((np.vectorize(u0))(self.x),dtype=complex)
        
        
    def step(self, renormalize = True):
        '''Make a step self.dt forward in time'''
        ufft = np.fft.fft(self.U)
        ufft *= np.exp((- self.D * self.k**2)* self.dt * (-1j))
        u_new = np.fft.ifft(ufft)
        u_new *= np.exp(-1j * self.V(self.x) * self.dt)
        if renormalize:
            u_new /= (np.sum(np.abs(u_new)**2*self.dx))**0.5
        self.U = u_new
   
    def plot(self):
        '''Plots the last step computed'''
        plt.plot(self.x, np.abs(U)**2)


## Definition of the problem

L = 20
tauMax = 10
J = 1000 #number of spatial points
N = 400 #number of temporal points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = 1
a=-1j #imaginary time


def V(x):
    """Potential of the BEC"""
    return 0.5*m*(w*x)**2


def u0(x):
    """initial state"""
    return np.exp(-10*(x)**2)

def advance(spectral, T):
    """Advances spectral of time T"""
    dt = spectral.dx**2
    for i in range(int(T/dt)):
        spectral.step((1j)*dt)
        #The factor should be dt for normal evolution
        # and 1j*dt for imaginary time evolution

sp = Spectral(L, tauMax, J, N, D, f, u0)


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


## Computation
def evolution_to_ground_anim():
    fig,ax = plt.subplots()
    line, = plt.plot([],[], label="$Wave \enspace function$")
    plt.xlim(-L/2, L/2)
    plt.ylim(-0.01,0.3)
    
    def make_frame(k):
        wave_function = np.abs(sp.U)**2
        line.set_data(sp.x, wave_function)
        #print(k)
        #ax.set_title("{:1.1e}".format(max(wave_function)))
        sp.step()
        return line,


    Vx = V(sp.x)
    plt.plot(sp.x, Vx, label="$Potential \enspace V$")
    
    E_n, psi_n = harmonic_state(0)
    plt.plot(sp.x,np.abs(psi_n)**2, label="$Analytical \enspace solution$")
    
    ani = animation.FuncAnimation(fig, make_frame, interval = 1, blit=False)
    
    plt.title("$Evolution \enspace process \enspace to \enspace get \enspace the \enspace background \enspace state$")
    plt.legend()
    plt.show()

evolution_to_ground_anim()