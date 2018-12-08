import numpy as np
import os

def get_ground(gS, threshold=1e-8):
    """Run the simulation until the imaginary time propagation is stable"""
    gS.renorm()
    gS.step()
    overlapping_previous_step = np.sum((np.abs(gS.oldU-gS.U))**2)
    ops0 = overlapping_previous_step
    gS.renorm()
    last_percentage_printed = 0
    while overlapping_previous_step>threshold:
        progress_percentage = int(100*(np.log(ops0)-np.log(overlapping_previous_step))/(np.log(ops0)-np.log(threshold)))
        if progress_percentage >= last_percentage_printed+10:
            print(str(progress_percentage)+"%")
            last_percentage_printed = progress_percentage
        gS.step()
        gS.renorm()
        overlapping_previous_step = np.sum((np.abs(gS.oldU-gS.U))**2)
    return gS.U
    
    
def save_ground(gS,src,overwrite=False,snapshot=False):
    name_func = "grd_state_2D_"+\
                str(int(gS.xMax))+"_"+\
                "{:1.1e}".format(gS.dtau)+"_"+\
                str(int(gS.Jx))+"_"+\
                "{:1.1e}".format(gS.pot.wx)+"_"+\
                "{:1.1e}".format(gS.pot.Ng)
                
    if overwrite or not os.path.isfile(src+"data/"+name_func+'.npy'):
        print("Saving the ground state...")
        if not snapshot:
            grd_wave_func = get_ground(gS)
        else:
            grd_wave_func = gS.U
        np.save(src+"data/"+name_func,grd_wave_func)


def load_ground(gS,src):
    name_func = "grd_state_2D_"+\
                str(int(gS.xMax))+"_"+\
                "{:1.1e}".format(gS.dtau)+"_"+\
                str(int(gS.Jx))+"_"+\
                "{:1.1e}".format(gS.pot.wx)+"_"+\
                "{:1.1e}".format(gS.pot.Ng)
    gS.U = np.load(src+"data/"+name_func+".npy")
    gS.oldU = np.copy(gS.U)
    