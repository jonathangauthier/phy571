import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import os

src = 'C:/Users/Maxime/Documents/phy571_project/'

os.chdir(src+'production/')
import vortex

def get_energy(gS,f,psi):
    """Get the energy of a state, and give the result of the hamiltonian over the state"""
    #calculate laplacian product
    hbar = gS.pot.hbar
    m = gS.pot.m
    Lx_psi = np.zeros((gS.Jx,gS.Jy),dtype=complex)
    Lx_psi[:,1:-1] = (psi[:,0:-2]+psi[:,2:]-2*psi[:,1:-1])/(gS.dx**2)
    Ly_psi = np.zeros((gS.Jx,gS.Jy),dtype=complex)
    Ly_psi[1:-1,:] = (psi[0:-2,:]+psi[2:,:]-2*psi[1:-1,:])/(gS.dx**2)
    H_psi =-(hbar)**2/(2*m)*Lx_psi -(hbar)**2/(2*m)*Ly_psi + 1j*hbar*f(gS.X,gS.Y,psi)
    E = np.abs(np.sum(np.conjugate(psi)*H_psi)*gS.dx*gS.dy)
    return E,H_psi

def evolution_anim(gS,number_of_steps_per_frame,isRenormed=False,saveMP4=False,frames=100,vtx=None):
    """Print the evolution of the BEC
            number_of_steps_per_frame: how many steps are calculated for each frame
            isRenormed: normalise the wave function at each step
            saveMP4: save the animation in a mp4 file with a number 'frames' of frames
    """
    fig,(ax1,ax2) = plt.subplots(ncols=2)

    
    #mesh1 = ax1.pcolormesh(gS.X,gS.Y,np.abs(gS.U)**2, norm=LogNorm(vmin=1e-4,vmax=1))
    mesh1 = ax1.pcolormesh(gS.X,gS.Y,np.abs(gS.U)**2)
    plt.colorbar(mesh1,ax=ax1)
    
    mesh2 = ax2.pcolormesh(gS.X,gS.Y,np.angle(gS.U), vmin=-np.pi, vmax=np.pi)
    plt.colorbar(mesh2,ax=ax2)
    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    if isRenormed:
        gS.renorm()
            
    def make_frame(k):
        C = (np.abs(gS.U)**2)[:-1,:-1]
        mesh1.set_array(C.ravel())
        
        C = (np.angle(gS.U))[:-1,:-1]
        mesh2.set_array(C.ravel())
        
        
        E,_ = get_energy(gS,gS.pot.f,gS.U)
        ax1.set_title("{:1.3e}".format(E))
            
        for i in range(number_of_steps_per_frame):
            gS.step()
            if isRenormed:
                gS.renorm()
            if vtx != None:
                vortex.force_phase(gS,vtx)
            
        return fig,
    
    ani = animation.FuncAnimation(fig, make_frame, frames = frames, blit=False)

    if saveMP4:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Mega Physicists'), bitrate=7200)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        name_func = "CN2D_"+\
                    str(int(gS.xMax))+"_"+\
                    "{:1.1e}".format(gS.dtau)+"_"+\
                    str(int(gS.Jx))+"_"+\
                    "{:1.1e}".format(gS.pot.wx)+"_"+\
                    "{:1.1e}".format(gS.pot.Ng)
                    
        if not os.path.isfile(src+"data\\"+name_func+".mp4"):
            ani.save(src+"data\\"+name_func+".mp4", writer=writer)
    else:
        plt.title("$Evolution \enspace process$")
        plt.show()
         
    
