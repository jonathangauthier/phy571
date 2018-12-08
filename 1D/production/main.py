import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as lin
from scipy.sparse import diags
import scipy.optimize as opt
import math
import os

src = 'C:/Users/Maxime/Documents/phy571_project/1D/'

os.chdir(src+'analysis/')
from compute_energy import get_energy

os.chdir(src+'design/')
from analytical_solution import harmonic_state
from crank_nicolson_1D import groundState1DCN
from potential import potential

os.chdir(src+'production/')
import ground_storage
import g_variation
from animation import evolution_anim



plt.rcParams['font.size'] = 22
    

## CASE 1 : Check algorithm with the ground state of the harmonic oscillator
"""
xMax = 30
dtau = 0.02
J = 1000 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = 0 #linear
pot = potential(w,Ng)
def u0(x):
    return np.exp(-10*(x)**2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,hbar=hbar,m=m)

n = 0
E_theo, psi = harmonic_state(gS,n)
E_calc,H_psi = get_energy(gS,psi)

fig,ax = plt.subplots()
wave_dens = plt.plot(gS.x,np.abs(psi)**2,label="harmonic ground state")
plt.ylabel('$|\psi|^2$')
theo_e = plt.plot(gS.x,[E_theo]*J,label="theoretical energy = "+"{:1.3e}".format(E_theo),linewidth=2)
calc_e = plt.plot(gS.x,[E_calc]*J,'--',label="calculated energy = "+"{:1.3e}".format(E_calc),linewidth=2)
plt.title("Calculated energy for the ground state of the linear equation for quantum oscillator")
plt.legend()
plt.show()
"""


## CASE 2 : repulsive BEC (g>0)
"""
xMax = 30
dtau = 0.02
J = 1000 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = 13
a=-1j #imaginary time
pot = potential(w,Ng)
def u0(x):
    return np.exp(-10*(x)**2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,a=a,hbar=hbar,m=m)
evolution_anim(gS,20,a=-1j,isRenormed=True)
"""


## CASE 3 : attractive BEC (g<0)
"""
xMax = 30
dtau = 0.02
J = 1000 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = -0.7
a=-1j #imaginary time
pot = potential(w,Ng)
def u0(x):
    return np.exp(-10*(x)**2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,a=a,hbar=hbar,m=m)
evolution_anim(gS,20,a=-1j,isRenormed=True)
"""


## CASE 4 : save ground
"""
xMax = 30
dtau = 0.02
J = 1000 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = -0.7
a=-1j #imaginary time
pot = potential(w,Ng)
def u0(x):
    return np.exp(-10*(x)**2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,a=a,hbar=hbar,m=m)
ground_storage.save_ground(gS,src)
"""


## CASE 5 : load ground
"""
xMax = 30
dtau = 0.02
J = 1000 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = -0.7
a=-1j #imaginary time
pot = potential(w,Ng)
def u0(x):
    return np.exp(-10*(x)**2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,a=a,hbar=hbar,m=m)
ground_storage.load_ground(gS,src)
fig,ax = plt.subplots()
wave_dens = plt.plot(gS.x,np.abs(gS.U)**2,label="Loaded ground state")
plt.ylabel('$|\psi|^2$')
plt.title("Loaded ground state from file")
plt.legend()
plt.show()
"""


## CASE 6 : energies for different Ng
"""
xMax = 10
dtau = 0.1
J = 500 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = -0.7
a=-1j #imaginary time
pot = potential(w,Ng)
def u0(x):
    alpha = m*w/hbar
    return (alpha/np.pi)**0.25*np.exp(-alpha*x**2/2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,a=a,hbar=hbar,m=m)
#g_variation.get_energies(gS,-5,5,21)
g_variation.plot_energies(gS,-5,5,21)
"""


## CASE 7 : standard deviations for different Ng
"""
xMax = 30
dtau = 0.02
J = 1000 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = -0.7
a=-1j #imaginary time
pot = potential(w,Ng)
def u0(x):
    return np.exp(-10*(x)**2)
gS = groundState1DCN(-xMax,xMax,dtau,J,D,pot,u0,a=a,hbar=hbar,m=m)
#g_variation.get_standard_deviations(gS,-0.3,0.3,21)
g_variation.plot_standard_deviations(gS,-0.3,0.3,21)
"""





