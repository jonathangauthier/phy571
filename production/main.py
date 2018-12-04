import numpy as np
import os
import matplotlib.pyplot as plt

src = 'C:/Users/Maxime/Documents/phy571_project/'

os.chdir(src+'design/')
from crank_nicolson_2D import groundState2DCN
from potential import potential

os.chdir(src+'production/')
from animation import evolution_anim
import ground_storage
import vortex


## Definition of the parameters

xMax = 10
dtau = 0.02
J = 1001 #number of spatial points
hbar = 1
m = 1
D = 1j*hbar/(2*m)
w = 0.1 #paramater of harmonic oscillator
Ng = 5
a=-1j


## Initialisation

pot = potential(w,w,Ng)

def u0(x,y):
    """initial state"""
    alpha = m*w/hbar
    return (alpha/np.pi)**0.5*np.exp(-alpha*(x**2+y**2)/2)

gS = groundState2DCN(-xMax,xMax,-xMax,xMax,dtau,J,J,D,pot,a)
gS.initialise_function(u0)


gS.U = np.load("ground_state_wave_function.npy")
gS.oldU = np.copy(gS.U)

## Computing

"""
ground_storage.save_ground(gS,src,overwrite=True)
ground_state = ground_storage.load_ground(gS,src)
"""

#evolution_anim(gS,20,isRenormed=True,saveMP4=False,frames=100)

#evolution_anim(gS,1,isRenormed=True,saveMP4=False,frames=100)

"""
a=1 #real time

gS.change_a(a)


evolution_anim(gS,20,isRenormed=False,saveMP4=False,frames=100)
"""




vtx = vortex.vortex(gS,xv=0,yv=0,xi=0.3,p=1)

vortex.shape_vortex(gS,vtx)

#evolution_anim(gS,1,isRenormed=False,vtx=vtx)

"""
#evolution_anim(gS,20)
"""