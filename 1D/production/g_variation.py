import numpy as np
import os
import matplotlib.pyplot as plt

src = 'C:/Users/Maxime/Documents/phy571_project/1D/'

os.chdir(src+'analysis/')
from compute_energy import get_energy

os.chdir(src+'design/')
from analytical_solution import harmonic_state
from crank_nicolson_1D import groundState1DCN
from potential import potential

os.chdir(src+'production/')
import ground_storage

def get_energies(gS0,Ng_min,Ng_max,Ng_nbr=11):
    """Save the energies for different values of Ng"""
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    E_array = np.zeros_like(Ng_array)
    for i, Ng in enumerate(Ng_array):
        pot = potential(gS0.pot.w,Ng)
        gS = groundState1DCN(-gS0.xMax,gS0.xMax,gS0.dtau,gS0.J,gS0.D,pot,gS0.u0)
        gd_wave_func = ground_storage.get_ground(gS)
        E,_ = get_energy(gS,gd_wave_func)
        E_array[i] = E
        print(i)
        
    np.save(src+"data\\"+"E_array",E_array)

def get_standard_deviation(gS,psi):
    return (np.sum(np.conjugate(psi)*gS.x**2*psi)*gS.dx-(np.sum(np.conjugate(psi)*gS.x*psi)*gS.dx)**2)**0.5

def get_standard_deviations(gS0,Ng_min,Ng_max,Ng_nbr=11):
    """Save the standard deviations for different values of Ng"""
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    std_array = np.zeros_like(Ng_array)
    for i, Ng in enumerate(Ng_array):
        pot = potential(gS0.pot.w,Ng)
        gS = groundState1DCN(-gS0.xMax,gS0.xMax,gS0.dtau,gS0.J,gS0.D,pot,gS0.u0)
        gd_wave_func = ground_storage.get_ground(gS)
        std = get_standard_deviation(gS,gd_wave_func)
        std_array[i] = std
        print(i)
       
    np.save(src+"data\\"+"std_array",std_array)


def plot_energies(gS,Ng_min,Ng_max,Ng_nbr=11):
    """Load the energies for different values of Ng and plot them"""
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    E_array = np.load(src+"data\\"+"E_array.npy")
    plt.plot(Ng_array,E_array,'.',label="ground state energies")
    E_harm,_ = harmonic_state(gS,0)
    plt.hlines(E_harm,Ng_min,Ng_max,label="Without non linearity")
    plt.xlabel("$Ng$")
    plt.ylabel("$E$")
    plt.title("$Ground \enspace state \enspace energies \enspace for \enspace different \enspace values \enspace of \enspace Ng$")
    plt.legend(loc="lower right")
    plt.show()
    
    
def plot_standard_deviations(gS,Ng_min,Ng_max,Ng_nbr=11):
    """Load the standard deviations for different values of Ng and plot them"""
    Ng_array = np.linspace(Ng_min,Ng_max,Ng_nbr)
    std_array = np.load(src+"data\\"+"std_array.npy")
    plt.plot(Ng_array,std_array,'.',label="ground state standard deviations")
    E_harm,psi_harm = harmonic_state(gS,0)
    plt.hlines(get_standard_deviation(gS,psi_harm),Ng_min,Ng_max,label="Without non linearity")
    plt.xlabel("$Ng$")
    plt.ylabel("$\sigma$")
    plt.title("$Ground \enspace state \enspace standard \enspace deviations \enspace for \enspace different \enspace values \enspace of \enspace Ng$")
    plt.legend(loc="upper left")
    plt.show()