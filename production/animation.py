import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import os

src = 'C:/Users/Maxime/Documents/phy571_project/'

os.chdir(src+'production/')
import vortex

def evolution_anim(gS,number_of_steps_per_frame,isRenormed=False,saveMP4=False,frames=100,vtx=None):
    """Print the evolution of the BEC
            number_of_steps_per_frame: how many steps are calculated for each frame
            isRenormed: normalise the wave function at each step
            saveMP4: save the animation in a mp4 file with a number 'frames' of frames
    """
    fig,_ = plt.subplots()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    mesh1 = ax1.pcolormesh(gS.X,gS.Y,np.abs(gS.U)**2, norm=LogNorm(vmin=1e-4,vmax=1))
    plt.colorbar(mesh1,ax=ax1)
    
    mesh2 = ax2.pcolormesh(gS.X,gS.Y,np.angle(gS.U), vmin=-np.pi, vmax=np.pi)
    plt.colorbar(mesh2,ax=ax2)
    
    if isRenormed:
        gS.renorm()
            
    def make_frame(k):
        C = (np.abs(gS.U)**2)[:-1,:-1]
        mesh1.set_array(C.ravel())
        
        C = (np.angle(gS.U))[:-1,:-1]
        mesh2.set_array(C.ravel())

        ax1.set_title("{:1.1e}".format(np.sum((np.abs(gS.oldU-gS.U))**2)))
            
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
         
    
