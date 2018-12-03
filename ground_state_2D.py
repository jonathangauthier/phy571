import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt
import math
import os.path
from mpl_toolkits import mplot3d
from matplotlib.colors import LogNorm

src = "C:\\Users\\Maxime\\Documents\\phy571_project\\"
os.chdir(src)

#from ground_state_1D import groundState1DCN


## Definition of the Algorithm

class groundState2DCN:
    """Find the ground state of the Gross-Pitaevski equation for a 2D problem via Crank-Nicholson"""
    
    def __init__(self,xMin,xMax, yMin,yMax, dtau, Jx, Jy, D, pot, u0, a=-1j):
        """Equation: du/dt = D*(d2u/dx2+d2/dy2) + pot.f(x,u)
                x in [xMin,xMax]
                y in [yMin, yMax]
                dtau: temporal step (dt = a*dtau)
                Jx spatial points in x
                Jy spatial points in y
                u(x,y,0) = u0(x,y)
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
        self.u0 = u0
        self.a = a
        
        #definition of the grid
        self.x, self.dx = np.linspace(xMin,xMax, Jx, retstep=True)
        self.y, self.dy = np.linspace(yMin,yMax, Jy, retstep=True)
        self.X, self.Y = np.meshgrid(self.x,self.y)
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
        diaUp = self.sigma_x*np.ones(self.Jx-1,dtype=complex)
        diaUp[0] = 0
        diaDown = self.sigma_x*np.ones(self.Jx-1,dtype=complex)
        diaDown[self.Jx-2] = 0
        dia = (1-2*self.sigma_x)*np.ones(self.Jx,dtype=complex)
        dia[0] = 1
        dia[-1] = 1
        self.Bx = diags([diaUp,dia,diaDown], [1,0,-1], (self.Jx, self.Jx), format='csr')
        
        #definition of the matrix By
        diaUp = self.sigma_y*np.ones(self.Jy-1,dtype=complex)
        diaUp[0] = 0
        diaDown = self.sigma_y*np.ones(self.Jy-1,dtype=complex)
        diaDown[self.Jy-2] = 0
        dia = (1-2*self.sigma_y)*np.ones(self.Jy,dtype=complex)
        dia[0] = 1
        dia[-1] = 1
        self.By = diags([diaUp,dia,diaDown], [1,0,-1], (self.Jy, self.Jy), format='csr')
        
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
        diaUp = self.sigma_x*np.ones(self.Jx-1,dtype=complex)
        diaUp[0] = 0
        diaDown = self.sigma_x*np.ones(self.Jx-1,dtype=complex)
        diaDown[self.Jx-2] = 0
        dia = (1-2*self.sigma_x)*np.ones(self.Jx,dtype=complex)
        dia[0] = 1
        dia[-1] = 1
        self.Bx = diags([diaUp,dia,diaDown], [1,0,-1], (self.Jx, self.Jx), format='csr')
        
        #redefinition of the matrix By
        diaUp = self.sigma_y*np.ones(self.Jy-1,dtype=complex)
        diaUp[0] = 0
        diaDown = self.sigma_y*np.ones(self.Jy-1,dtype=complex)
        diaDown[self.Jy-2] = 0
        dia = (1-2*self.sigma_y)*np.ones(self.Jy,dtype=complex)
        dia[0] = 1
        dia[-1] = 1
        self.By = diags([diaUp,dia,diaDown], [1,0,-1], (self.Jy, self.Jy), format='csr')
        
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
        
    def renorm(self,vortex=False):
        """Renormalize the current solution"""
        self.U /= (np.sum(np.abs(self.U)**2)*self.dx*self.dy)**0.5
        if vortex:
            self.U = np.abs(self.U)*np.exp(1j*np.angle(self.X+1j*self.Y))
        


class Potential:
    """Specific class to define potential and its parameters"""
    
    def __init__(self,wx,wy,Ng,m=1,hbar=1):
        self.wx = wx
        self.wy = wy
        self.Ng = Ng
        self.m = m
        self.hbar = hbar
        
        def V(x,y):
            """Potential of the BEC"""
            return 0.5*self.m*((self.wx*x)**2+(self.wy*y)**2)
            
        def Veff(x,y,u):
            """Effective potential of the BEC"""
            return V(x,y)+ self.Ng*np.abs(u)**2
            
        def f(x,y,u):
            return 1/(1j*self.hbar)*Veff(x,y,u)*u
          
        self.V = V
        self.Veff = Veff    
        self.f = f
     

## Definition of the problem

xMax = 10
dtau = 0.02
J = 201 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = 5
a=-1j #imaginary time


pot = Potential(w,w,Ng)

def u0(x,y):
    """initial state"""
    alpha = m*w/hbar
    return (alpha/np.pi)**0.5*np.exp(-alpha*(x**2+y**2)/2)
    
gS = groundState2DCN(-xMax,xMax,-xMax,xMax,dtau,J,J,D,pot,u0)


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


def evolution_anim(number_of_steps_per_frame, potential_size_factor=50,a=-1j,isRenormed=True,shape='contours',saveMP4=False,plotAnalytical=True,vortex=False):
    """Print the imaginary time evolution of the BEC to the ground state
            number_of_steps_per_frame: how many steps are calculated for each frame
            potential_size_factor: the potential is divided by this value to fit in the window
    """
    gS.change_a(a)
    
    fig,ax = plt.subplots()
    fig.set_size_inches(16/9*7,7)
    ax = fig.add_subplot(111, projection='3d')
    opacity = 1
    
    if shape=='surface' and plotAnalytical:
        ax.view_init(azim=45, elev=-5)
        opacity = 0.6
    
    E, psi = harmonic_state_2D(0,0)
    
    
    def make_frame(k):
        if isRenormed:
            gS.renorm(vortex)
        
        Veff_current = (np.vectorize(gS.pot.Veff))(gS.X,gS.Y, gS.U)
        Veff_current /= potential_size_factor
         
        ax.clear()
        
        if shape == 'surface':
            density_proba = ax.plot_surface(gS.X,gS.Y,np.abs(gS.U)**2, label="$Density \enspace probability$",cmap='winter',alpha=opacity)
            if plotAnalytical:
                analytical = ax.plot_surface(gS.X,gS.Y,np.abs(psi)**2,rstride=1,label="$Analytical \enspace solution \enspace level \enspace 0$",cmap='autumn',alpha=0.6)
            #potential = ax.plot_surface(gS.X,gS.Y,gS.pot.V(gS.X,gS.Y)/potential_size_factor,label="$Potential \enspace V$",color=(0.5,0,0,0.5))
            #effective_potential = ax.plot_surface(gS.X,gS.Y,Veff_current,label="$Effective \enspace potential \enspace Veff$")
        elif shape == 'contours':
            density_proba = ax.contour3D(gS.X, gS.Y, np.abs(gS.U)**2, 10, cmap='Blues')
            if plotAnalytical:
                analytical = ax.contour3D(gS.X,gS.Y,np.abs(psi)**2,10,label="$Analytical \enspace solution \enspace level \enspace 0$",cmap='Reds')

        ax.set_title("{:1.1e}".format(np.sum((np.abs(gS.oldU-gS.U))**2)))
        ax.set_xlim3d(gS.xMin,gS.xMax)
        ax.set_ylim3d(gS.yMin,gS.yMax)
        ax.set_zlim3d(-0.01,0.02)
            
        for i in range(number_of_steps_per_frame):
            if isRenormed:
                gS.renorm(vortex)
            gS.step()

        return density_proba,
    

    ani = animation.FuncAnimation(fig, make_frame, frames = 70, interval = number_of_steps_per_frame*10, blit=False)

    if saveMP4:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=7200)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        ani.save(src+'CN2D.mp4', writer=writer)
    else:
        plt.title("$Evolution \enspace process \enspace to \enspace get \enspace the \enspace ground \enspace state$")
        plt.show()

def evolution_anim_color(number_of_steps_per_frame, potential_size_factor=50,a=-1j,isRenormed=True,shape='contours',saveMP4=False,plotAnalytical=True,vortex=False,xv=0,yv=0):
    """Print the imaginary time evolution of the BEC to the ground state
            number_of_steps_per_frame: how many steps are calculated for each frame
            potential_size_factor: the potential is divided by this value to fit in the window
    """
    gS.change_a(a)
    
    fig,ax = plt.subplots()

    mesh = ax.pcolormesh(gS.X,gS.Y,np.zeros_like(gS.X), norm=LogNorm(vmin=0.0001,vmax=1))
    plt.colorbar(mesh,ax=ax)

    
    def make_frame(k):
        if isRenormed:
            gS.renorm(vortex)
        
        C = (np.abs(gS.U)**2)[:-1,:-1]
        mesh.set_array(C.ravel())

        ax.set_title("{:1.1e}".format(np.sum((np.abs(gS.oldU-gS.U))**2)))
            
        for i in range(number_of_steps_per_frame):
            if isRenormed:
                gS.renorm(vortex)
            gS.step()
            
        return mesh,
    
    ani = animation.FuncAnimation(fig, make_frame, blit=False)
    plt.show()

def get_energy(f,psi):
    """Get the energy of a state, and give the result of the hamiltonian over the state"""
    #calculate laplacian product
    Lx_psi = np.zeros((gS.Jx,gS.Jy),dtype=complex)
    Lx_psi[:,1:-1] = (psi[:,0:-2]+psi[:,2:]-2*psi[:,1:-1])/(gS.dx**2)
    Ly_psi = np.zeros((gS.Jx,gS.Jy),dtype=complex)
    Ly_psi[1:-1,:] = (psi[0:-2,:]+psi[2:,:]-2*psi[1:-1,:])/(gS.dx**2)
    H_psi =-(hbar)**2/(2*m)*Lx_psi -(hbar)**2/(2*m)*Ly_psi + 1j*hbar*np.vectorize(f)(gS.X,gS.Y,psi)
    E = np.sum(np.conjugate(psi)*H_psi)*gS.dx*gS.dy
    return E,H_psi


def get_ground(gS, threshold=1e-8):
    gS.renorm()
    gS.step()
    while np.sum((np.abs(gS.oldU-gS.U))**2)>threshold:
        gS.renorm()
        gS.step()
        gS.renorm()
    return gS.U
    
    
def save_ground(gS,overwrite=False):
    name_func = "grd_state_2D_"+\
                str(int(gS.xMax))+"_"+\
                "{:1.1e}".format(gS.dtau)+"_"+\
                str(int(gS.Jx))+"_"+\
                "{:1.1e}".format(gS.pot.wx)+"_"+\
                "{:1.1e}".format(gS.pot.Ng)
                
    if overwrite or not os.path.isfile(src+"data\\"+name_func):
        grd_wave_func = get_ground(gS)
        np.savetxt(src+"data\\"+name_func+".txt",grd_wave_func)

def load_ground(gS):
    name_func = "grd_state_2D_"+\
                str(int(gS.xMax))+"_"+\
                "{:1.1e}".format(gS.dtau)+"_"+\
                str(int(gS.Jx))+"_"+\
                "{:1.1e}".format(gS.pot.wx)+"_"+\
                "{:1.1e}".format(gS.pot.Ng)
    gS.U = np.loadtxt(src+"data\\"+name_func+".txt",dtype=complex)
    
    
## Evolution

#Animation of the evolution


#evolution_anim_color(10,10,a=-1j,isRenormed=True,vortex=False)
#evolution_anim_color(10,10,a=1,isRenormed=False,vortex=True)


#evolution_anim(10,10,a=-1j,isRenormed=True,shape='surface',plotAnalytical=False,vortex=True)
#evolution_anim(10,10,a=1,isRenormed=False,shape='surface',plotAnalytical=False,vortex=False)



#evolution_anim(20,10,a=1,isRenormed=False)

"""
xv_coord = int((xv-gS.xMin)//gS.dx + ((xv-gS.xMin)%gS.dx>0.5))+1
yv_coord = int((yv-gS.yMin)//gS.dy + ((yv-gS.yMin)%gS.dy>0.5))+1
"""


save_ground(gS)
load_ground(gS)







