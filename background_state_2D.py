import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt
import math
from  scipy.misc import derivative as der
import os.path
from mpl_toolkits import mplot3d


src = "C:\\Users\\Maxime\\Documents\\phy571_project\\"
os.chdir(src)
#from background_state_1D import groundState1DCN



## Definition of the Algorithm

class groundState2DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 1D problem via Crank-Nicholson"""
    
    def __init__(self,xMin,xMax, yMin,yMax, tauMax, Jx, Jy, N, D, pot, u0, a=-1j):
        """Equation: du/dt = D*(d2u/dx2+d2/dy2) + pot.f(x,u)
                x in [xMin,xMax]
                y in [yMin, yMax]
                t in [0,-1j*tauMax]
                Jx spatial points in x
                Jy spatial points in y
                N temporal points
                u(x,y,0) = u0(x,y)
           Evolve in imaginary time for a=-1j and in real time for a=1
        """
        #definition of parameters
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.tauMax = tauMax
        self.Jx = Jx
        self.Jy = Jy
        self.N = N
        self.D = D
        self.pot = pot
        self.u0 = u0
        self.a = a
        
        #definition of the grid
        self.x, self.dx = np.linspace(xMin,xMax, Jx, retstep=True)
        self.y, self.dy = np.linspace(yMin,yMax, Jy, retstep=True)
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.tau, self.dtau = np.linspace(0,tauMax, N, retstep=True)
        self.dt = a*self.dtau
        self.sigma_x = D*self.dt/(2*self.dx**2)
        self.sigma_y = D*self.dt/(2*self.dy**2)
        
        #definition of the matrix Ax
        Ax = np.zeros((3,Jx),dtype=complex)
        Ax[0,1] = 0
        Ax[0,2:] = -self.sigma_x
        Ax[1,:] = 1+2*self.sigma_x
        Ax[1,0] = 1
        Ax[1,-1] = 1
        Ax[2,:-2] = -self.sigma_x
        Ax[2,-2] = 0
        self.Ax = Ax
        
        #definition of the matrix Ay
        Ay = np.zeros((3,Jy),dtype=complex)
        Ay[0,1] = 0
        Ay[0,2:] = -self.sigma_y
        Ay[1,:] = 1+2*self.sigma_y
        Ay[1,0] = 1
        Ay[1,-1] = 1
        Ay[2,:-2] = -self.sigma_y
        Ay[2,-2] = 0
        self.Ay = Ay
        
        #definition of the matrix Bx
        diaUp = [self.sigma_x]*(self.Jx-1)
        diaUp[0] = 0
        diaDown = [self.sigma_x]*(self.Jx-1)
        diaDown[Jx-2] = 0
        dia = [1-2*self.sigma_x]*self.Jx
        dia[0] = 1
        dia[-1] = 1
        diagonals = [dia, diaUp, diaDown]
        self.Bx = np.array(diags(diagonals, [0, 1, -1]).toarray(),dtype=complex)
        
        #definition of the matrix By
        diaUp = [self.sigma_y]*(self.Jy-1)
        diaUp[0] = 0
        diaDown = [self.sigma_y]*(self.Jy-1)
        diaDown[Jy-2] = 0
        dia = [1-2*self.sigma_y]*self.Jy
        dia[0] = 1
        dia[-1] = 1
        diagonals = [dia, diaUp, diaDown]
        self.By = np.array(diags(diagonals, [0, 1, -1]).toarray(),dtype=complex)
        
        #definition of the initial condition
        self.U = np.array((np.vectorize(u0))(self.X,self.Y),dtype=complex)
        self.oldU = np.zeros_like(self.U, dtype=complex)
        self.isFirstStep = True
    
    def change_a(self,a):
        self.a = a
        
        self.dt = a*self.dtau
        self.sigma_x = D*self.dt/(2*self.dx**2)
        self.sigma_y = D*self.dt/(2*self.dy**2)
        
        #redefinition of the matrix Ax
        Ax = np.zeros((3,self.Jx),dtype=complex)
        Ax[0,1] = 0
        Ax[0,2:] = -self.sigma_x
        Ax[1,:] = 1+2*self.sigma_x
        Ax[1,0] = 1
        Ax[1,-1] = 1
        Ax[2,:-2] = -self.sigma_x
        Ax[2,-2] = 0
        self.Ax = Ax
        
        #redefinition of the matrix Ay
        Ay = np.zeros((3,self.Jy),dtype=complex)
        Ay[0,1] = 0
        Ay[0,2:] = -self.sigma_y
        Ay[1,:] = 1+2*self.sigma_y
        Ay[1,0] = 1
        Ay[1,-1] = 1
        Ay[2,:-2] = -self.sigma_y
        Ay[2,-2] = 0
        self.Ay = Ay
        
        #redefinition of the matrix Bx
        diaUp = [self.sigma_x]*(self.Jx-1)
        diaUp[0] = 0
        diaDown = [self.sigma_x]*(self.Jx-1)
        diaDown[self.Jx-2] = 0
        dia = [1-2*self.sigma_x]*self.Jx
        dia[0] = 1
        dia[-1] = 1
        diagonals = [dia, diaUp, diaDown]
        self.Bx = np.array(diags(diagonals, [0, 1, -1]).toarray(),dtype=complex)
        
        #redefinition of the matrix By
        diaUp = [self.sigma_y]*(self.Jy-1)
        diaUp[0] = 0
        diaDown = [self.sigma_y]*(self.Jy-1)
        diaDown[self.Jy-2] = 0
        dia = [1-2*self.sigma_y]*self.Jy
        dia[0] = 1
        dia[-1] = 1
        diagonals = [dia, diaUp, diaDown]
        self.By = np.array(diags(diagonals, [0, 1, -1]).toarray(),dtype=complex)
        
    def step(self):
        """Calculate the next step in the Crank-Nicholson algorithm"""
        #for the first step, we calculate only with the previous state
        if self.isFirstStep:
            F = (np.vectorize(self.pot.f))(self.x, self.y, self.U)
            self.isFirstStep = False
        #else, we calculate with the previous step and the one before to keep the second order accuracy
        else:
            F = 3/2*(np.vectorize(self.pot.f))(self.x, self.y, self.U)-1/2*(np.vectorize(self.pot.f))(self.x, self.y, self.oldU)
        self.oldU = np.copy(self.U)
        #Propagation in x
        C = np.dot(self.Bx,self.U)+0.5*self.dt*F
        C[0,:] = 0
        C[-1,:] = 0
        self.U = lin.solve_banded((1,1), self.Ax, C)
        #Propagation in y
        C = np.dot(self.By,self.U.T)+0.5*self.dt*F
        C[0,:] = 0
        C[-1,:] = 0
        self.U = lin.solve_banded((1,1), self.Ay, C).T
        
    def renorm(self):
        """Renormalize the current solution"""
        self.U /= (np.sum(np.abs(self.U)**2)*self.dx*self.dy)**0.5
        


class Potential:
    """Specific class to define potential and its parameters"""
    
    def __init__(self,wx,wy,Ng,m=1,hbar=1):
        self.wx = wx
        self.wy = wy
        self.Ng = Ng
        self.m = 1
        self.hbar = 1
        
        def V(x,y):
            """Potential of the BEC"""
            return 0.5*self.m*((self.wx*x)**2+(self.wy*y)**2)
            
        def Veff(x,y,u):
            """Effective potential of the BEC"""
            return V(x,y)+ 0*self.Ng*np.abs(u)**2
            
        def f(x,y,u):
            return 1/(1j*self.hbar)*Veff(x,y,u)*u
          
        self.V = V
        self.Veff = Veff    
        self.f = f
     

## Definition of the problem

xMax = 5
tauMax = 10
J = 50 #number of spatial points
N = 400 #number of temporal points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = +1000*0.05/10
a=-1j #imaginary time


pot = Potential(w,w,Ng)

def u0(x,y):
    """initial state"""
    return np.exp(-10*(x**2+y**2))
    
gS = groundState2DCN(-xMax,xMax,-xMax,xMax,tauMax,J,J,N,D,pot,u0)


## Definition of the useful functions

def hermite_poly(x,n):
    """Give the value of the n-th physicists' Hermite polynomial at x"""
    coef = [0]*(n+1)
    coef[-1]=1
    return np.polynomial.hermite.hermval(x,coef)


def harmonic_state_2D(nx,ny):
    """Give the energy and the function corresponding of the linear harmonic oscillator in the nx-th state in x and in ny-th state in y"""
    E_nx = hbar*gS.pot.wx*(nx+1/2)
    E_ny = hbar*gS.pot.wy*(ny+1/2)
    E = E_nx + E_ny
    psi = (2**nx*math.factorial(nx))**(-0.5)*(m*gS.pot.wx/(np.pi*hbar))**0.25*np.exp(-m*gS.pot.wx*gS.X**2/(2*hbar))*\
               hermite_poly((m*gS.pot.wx/hbar)**0.5*gS.X,nx) *\
               (2**ny*math.factorial(ny))**(-0.5)*(m*gS.pot.wy/(np.pi*hbar))**0.25*np.exp(-m*gS.pot.wy*gS.Y**2/(2*hbar))*\
               hermite_poly((m*gS.pot.wy/hbar)**0.5*gS.Y,ny)
    return E, psi


def save_simulation():
    for k in range(50):
        gS.renorm()
        gS.step()
        
    np.savetxt(src+"data\\"+"bgFunc.txt",gS.U)


def evolution_to_ground_static():
    for k in range(N):
        gS.renorm()
        plt.plot(gS.x, gS.U)
        gS.step()
        
    plt.show()


def evolution_anim(number_of_steps_per_frame, potential_size_factor=50,a=-1j,isRenormed=True):
    """Print the imaginary time evolution of the BEC to the ground state
            number_of_steps_per_frame: how many steps are calculated for each frame
            potential_size_factor: the potential is divided by this value to fit in the window
    """
    gS.change_a(a)
    
    fig,ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(gS.xMin,gS.xMax)
    ax.set_ylim3d(gS.yMin,gS.yMax)
    ax.set_zlim3d(-0.01,0.1)
    
    
    surf_wave_function = ax.plot_surface(gS.X,gS.Y,np.zeros_like(gS.X), label="$Wave \enspace function$",color=(0,0,0.5,0.5))
    surf_potential = ax.plot_surface(gS.X,gS.Y,gS.pot.V(gS.X,gS.Y)/potential_size_factor,label="$Potential \enspace V$",color=(0.5,0,0,0.5))
    #surf_effective_potential = ax.plot_surface(gS.X,gS.Y,np.zeros_like(gS.X),label="$Effective \enspace potential \enspace Veff$")
    
    
    E, psi = harmonic_state_2D(0,0)
    
    def make_frame(k):
        if isRenormed:
            gS.renorm()
        
        Veff_current = (np.vectorize(gS.pot.Veff))(gS.X,gS.Y, gS.U)
        Veff_current /= potential_size_factor
            
        ax.clear()    
        surf_wave_function = ax.plot_surface(gS.X,gS.Y,np.abs(gS.U)**2, label="$Wave \enspace function$",color=(0,0,0.5,0.5))
        ax.plot_surface(gS.X,gS.Y,np.abs(psi)**2,rstride=1,label="$Analytical \enspace solution \enspace level \enspace 0$",color=(0,0.5,0,0.5))
        surf_potential = ax.plot_surface(gS.X,gS.Y,gS.pot.V(gS.X,gS.Y)/potential_size_factor,label="$Potential \enspace V$",color=(0.5,0,0,0.5))
        #surf_effective_potential = ax.plot_surface(gS.X,gS.Y,Veff_current,label="$Effective \enspace potential \enspace Veff$")
        ax.set_title("{:1.1e}".format(np.sum((np.abs(gS.oldU-gS.U))**2)))
        ax.set_xlim3d(gS.xMin,gS.xMax)
        ax.set_ylim3d(gS.yMin,gS.yMax)
        ax.set_zlim3d(-0.01,0.1)    
            
        for i in range(number_of_steps_per_frame):
            if isRenormed:
                gS.renorm()
            gS.step()

        return surf_wave_function,
        
    
    ani = animation.FuncAnimation(fig, make_frame, interval = 20, blit=False)
    
    plt.title("$Evolution \enspace process \enspace to \enspace get \enspace the \enspace ground \enspace state$")
    #plt.legend()
    plt.show()


def get_energy(f,psi):
    """Get the energy of a state, and give the result of the hamiltonian over the state"""
    #calculate laplacian product
    L_psi = np.zeros(J)
    L_psi[1:-1] = (psi[0:-2]+psi[2:]-2*psi[1:-1])/(gS.dx**2)
    H_psi =-(hbar)**2/(2*m)*L_psi + 1j*hbar*np.vectorize(f)(gS.x,psi)
    E = np.sum(np.conjugate(psi)*H_psi)*gS.dx
    return E,H_psi


def get_ground(gS, threshold=1e-7):
    gS.renorm()
    gS.step()
    while np.sum((np.abs(gS.oldU-gS.U))**2)>threshold:
        gS.renorm()
        gS.step()
        gS.renorm()
    return gS.U
    
    

def get_energies(Ng_min,Ng_max,Ng_nbr=11):
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    E_array = np.zeros_like(Ng_array)
    for i, Ng in enumerate(Ng_array):
        def V(x):
            """Potential of the BEC"""
            return 0.5*m*(w*x)**2
            
        def Veff(x,u):
            """Effective potential of the BEC"""
            return V(x)+ Ng*np.abs(u)**2
            
        def f(x,u):
            return 1/(1j*hbar)*Veff(x,u)*u
            
        gS = groundState1DCN(-xMax,xMax,tauMax,J,N,D,f,u0)
        
        bkgd_wave_func = get_ground(gS)
        E,H_psi = get_energy(f,bkgd_wave_func)
        E_array[i] = E
        print(i)
        
    np.savetxt(src+"data\\"+"E_array.txt",E_array)

def get_standard_deviation(psi):
    return (np.sum(np.conjugate(psi)*gS.x**2*psi)*gS.dx-(np.sum(np.conjugate(psi)*gS.x*psi)*gS.dx)**2)**0.5

def get_standard_deviations(Ng_min,Ng_max,Ng_nbr=11):
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    std_array = np.zeros_like(Ng_array)
    for i, Ng in enumerate(Ng_array):
        def V(x):
            """Potential of the BEC"""
            return 0.5*m*(w*x)**2
            
        def Veff(x,u):
            """Effective potential of the BEC"""
            return V(x)+ Ng*np.abs(u)**2
            
        def f(x,u):
            return 1/(1j*hbar)*Veff(x,u)*u
            
        gS = groundState1DCN(-xMax,xMax,tauMax,J,N,D,f,u0)
        
        bkgd_wave_func = get_ground(gS)
        std = get_standard_deviation(bkgd_wave_func)
        std_array[i] = std
        print(i)
       
    np.savetxt(src+"data\\"+"std_array.txt",std_array)


def plot_energies(Ng_min,Ng_max,Ng_nbr=11):
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    E_array = np.loadtxt(src+"data\\"+"E_array.txt",dtype=complex)
    plt.plot(Ng_array,E_array,'.')
    E_harm,_ = harmonic_state(0)
    plt.hlines(E_harm,Ng_min,Ng_max,label="Without non linearity")
    plt.xlabel("$Ng$")
    plt.ylabel("$E$")
    plt.title("$Ground \enspace state \enspace energies \enspace for \enspace different \enspace values \enspace of \enspace Ng$")
    plt.legend(loc="lower right")
    plt.show()
    
    
def plot_standard_deviations(Ng_min,Ng_max,Ng_nbr=11):
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    std_array = np.loadtxt(src+"data\\"+"std_array.txt",dtype=complex)
    plt.plot(Ng_array,std_array,'.')
    E_harm,psi_harm = harmonic_state(0)
    plt.hlines(get_standard_deviation(psi_harm),Ng_min,Ng_max,label="Without non linearity")
    plt.xlabel("$Ng$")
    plt.ylabel("$\sigma$")
    plt.title("$Ground \enspace state \enspace standard \enspace deviations \enspace for \enspace different \enspace values \enspace of \enspace Ng$")
    plt.legend(loc="upper right")
    plt.show()
    

def save_ground(gS,overwrite=False):
    name_func = "grd_state_"+str(int(gS.xMax))+"_"+str(int(gS.tauMax))+"_"+str(int(gS.J))+"_"+str(int(gS.N))+\
                "_"+"{:1.1e}".format(gS.pot.w)+"_"+str(int(200*gS.pot.Ng))
                
    if overwrite or not os.path.isfile(src+"data\\"+name_func):
        grd_wave_func = get_ground(gS)
        np.savetxt(src+"data\\"+name_func,grd_wave_func)


def load_ground(gS):
    name_func = "grd_state_"+str(int(gS.xMax))+"_"+str(int(gS.tauMax))+"_"+str(int(gS.J))+"_"+str(int(gS.N))+\
                "_"+"{:1.1e}".format(gS.pot.w)+"_"+str(int(200*gS.pot.Ng))
    gS.U = np.loadtxt(src+"data\\"+name_func,dtype=complex)
    
    

    
## Evolution

#Animation of the evolution

evolution_anim(20,10,a=-1j,isRenormed=True)



#Check with the ground state of the harmonic oscillator
"""
n = 0
E_theo, psi = harmonic_state(n)
E_calc,H_psi = get_energy(lambda x,u: 1/(1j*hbar)*V(x)*u,psi)

plt.plot(gS.x,np.abs(psi)**2,label="harmonic eigen state")
plt.plot(gS.x,[E_theo]*J,label="theoretical energy")
plt.plot(gS.x,[E_calc]*J,label="calculated energy")
plt.legend()
plt.show()
"""


#save ground
"""
bkgd_wave_func = get_ground()
np.savetxt(src+"data\\"+"bkgd_wave_func.txt",bkgd_wave_func)
"""

#load ground
"""
bkgd_wave_func = np.loadtxt(src+"data\\"+"bkgd_wave_func.txt",dtype=complex)
plt.plot(np.abs(a)**2)
plt.plot(np.abs(bkgd_wave_func)**2)
plt.show()
"""

"""
#get_energies(-2000*0.05/10,2000*0.05/10)
plot_energies(-2000*0.05/10,2000*0.05/10)
"""

"""
#get_standard_deviations(-2000*0.05/10,2000*0.05/10)
plot_standard_deviations(-2000*0.05/10,2000*0.05/10)
"""

"""
save_ground(gS)
load_ground(gS)

grd_wave_func = gS.U


plt.plot(gS.x,np.abs(grd_wave_func)**2)
plt.show()
"""

#evolution_anim(20,10,a=1,isRenormed=False)


