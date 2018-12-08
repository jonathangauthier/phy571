import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

src = 'C:/Users/Maxime/Documents/phy571_project/1D/'

os.chdir(src+'design/')
from analytical_solution import harmonic_state

os.chdir(src+'analysis/')
from compute_energy import get_energy


def evolution_anim(gS,number_of_steps_per_frame,a=-1j,isRenormed=True):
    """Print the imaginary time evolution of the BEC to the ground state
            number_of_steps_per_frame: how many steps are calculated for each frame
            potential_size_factor: the potential is divided by this value to fit in the window
    """
    fig,ax = plt.subplots()
    fig.set_size_inches(16/9*7,7)
    line_wave_function, = ax.plot([],[], label="$Wave \enspace function$", color='dodgerblue')
    E_n, psi_n = harmonic_state(gS,0)
    line_solution, = ax.plot(gS.x,np.abs(psi_n)**2, label="$Analytical \enspace solution \enspace level \enspace 0$", color='orange')
    plt.xlim(-gS.xMax,gS.xMax)
    plt.ylim(-0.01,0.5)
    plt.xlabel('Probability density for the BEC ground state')
    plt.ylabel('$|\psi|^2$')
    
    gS.change_a(a)
    
    Vx = gS.pot.V(gS.x)
    ax2 = ax.twinx()
    line_potential, = ax2.plot(gS.x, Vx, label="$Potential \enspace V$", color='orangered')
    line_effective_potential, = ax2.plot([],[], label="$Effective \enspace potential \enspace V_{eff}$", color='forestgreen')
    plt.ylabel('$V \enspace , \enspace V_{eff}$')

    def make_frame(k):
        if isRenormed:
            gS.renorm()
        line_wave_function.set_data(gS.x, np.abs(gS.U)**2)
        
        
        Veff_current = (gS.pot.Veff)(gS.x, gS.U)
        line_effective_potential.set_data(gS.x, Veff_current)
        
        plt.legend(handles=[line_wave_function,line_solution,line_potential,line_effective_potential])
        E,_ = get_energy(gS,gS.U)
        ax.set_title("Energy : "+"{:1.3e}".format(E))
        
        for i in range(number_of_steps_per_frame):
            if isRenormed:
                gS.renorm()
            gS.step()
        return line_wave_function,
        
    
    plt.xlabel('Probability density for the BEC ground state')
    ani = animation.FuncAnimation(fig, make_frame, interval = 20, blit=False)
    
    plt.show()