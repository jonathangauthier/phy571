import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20


src = 'C:/Users/Maxime/Documents/phy571_project/2D/'

os.chdir(src+'design/')
from crank_nicolson_2D import groundState2DCN
from potential import potential

os.chdir(src+'production/')
import ground_storage
from animation import evolution_anim
import vortex


## CASE 1 : Check algorithm with the ground state of the harmonic oscillator
"""
xMax = 10
dtau = 0.01
J = 1001 #number of spatial points in one direction
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.2 #paramater of harmonic oscillator
Ng = 0 #No non-linear term -> Schrödinger equation
a=-1j
pot = potential(w,w,Ng)
def u0(x,y):
    #initial state (we start from the solution of the Schrödinger equation to be close to the ground state)
    alpha = m*w/hbar
    return (alpha/np.pi)**0.5*np.exp(-alpha*(x**2+y**2)/2)
gS = groundState2DCN(-xMax,xMax,-xMax,xMax,dtau,J,J,D,pot,a)
gS.initialise_function(u0)
#since we start form the analytical solution, the wave function should not evolve
evolution_anim(gS,1,isRenormed=True)
"""


## CASE 2 : Ground state for g>0 (repulsive BEC)
"""
xMax = 10
dtau = 0.5
J = 501 #number of spatial points in one direction
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.2 #paramater of harmonic oscillator
Ng = 10 #No non-linear term -> Schrödinger equation
a=-1j
pot = potential(w,w,Ng)
def u0(x,y):
    #initial state (we start from the solution of the Schrödinger equation to be close to the ground state)
    alpha = m*w/hbar
    return (alpha/np.pi)**0.5*np.exp(-alpha*(x**2+y**2)/2)
gS = groundState2DCN(-xMax,xMax,-xMax,xMax,dtau,J,J,D,pot,a)
gS.initialise_function(u0)
evolution_anim(gS,1,isRenormed=True)
"""


## CASE 3 : Ground state for g<0 (attractive BEC)
"""
xMax = 10
dtau = 0.5
J = 501 #number of spatial points in one direction
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.2 #paramater of harmonic oscillator
Ng = -3 #No non-linear term -> Schrödinger equation
a=-1j
pot = potential(w,w,Ng)
def u0(x,y):
    #initial state (we start from the solution of the Schrödinger equation to be close to the ground state)
    alpha = m*w/hbar
    return (alpha/np.pi)**0.5*np.exp(-alpha*(x**2+y**2)/2)
gS = groundState2DCN(-xMax,xMax,-xMax,xMax,dtau,J,J,D,pot,a)
gS.initialise_function(u0)
evolution_anim(gS,1,isRenormed=True)
"""


## CASE 4 : Simulating a vortex
"""
xMax = 100
dtau = 0.5
J = 501 #number of spatial points in one direction
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.001 #paramater of harmonic oscillator
Ng = 5 #No non-linear term -> Schrödinger equation
a=-1j
pot = potential(w,w,Ng)
def u0(x,y):
    #initial state (we start from the solution of the Schrödinger equation to be close to the ground state)
    alpha = m*w/hbar
    return (alpha/np.pi)**0.5*np.exp(-alpha*(x**2+y**2)/2)
gS = groundState2DCN(-xMax,xMax,-xMax,xMax,dtau,J,J,D,pot,a)
gS.initialise_function(u0)


ground_storage.save_ground(gS,src)
ground_storage.load_ground(gS,src)


vtx = vortex.vortex(gS,xv=0,yv=0,xi=1,p=1)
vortex.shape_vortex(gS,vtx)
vortex.force_phase(gS,vtx)


gS.dtau = 0.1
gS.change_a(1)
evolution_anim(gS,1,isRenormed=False,vtx=vtx)
"""

