import numpy as np
import os

def get_ground(gS, threshold=1e-7):
    """Run the simulation until the imaginary time propagation is stable"""
    gS.change_a(-1j) #imaginary time
    gS.renorm()
    gS.step()
    while np.sum((np.abs(gS.oldU-gS.U))**2)>threshold:
        gS.renorm()
        gS.step()
        gS.renorm()
    return gS.U
    
def save_ground(gS,src,overwrite=False):
    name_func = "grd_state_"+str(int(gS.xMax))+"_"+"{:1.3e}".format(gS.dtau)+"_"+str(int(gS.J))+\
                "_"+"{:1.3e}".format(gS.pot.w)+"_"+"{:1.3e}".format(gS.pot.Ng)
                
    if overwrite or not os.path.isfile(src+"data\\"+name_func):
        grd_wave_func = get_ground(gS)
        np.save(src+"data/"+name_func,grd_wave_func)


def load_ground(gS,src):
    name_func = "grd_state_"+str(int(gS.xMax))+"_"+"{:1.3e}".format(gS.dtau)+"_"+str(int(gS.J))+\
                "_"+"{:1.3e}".format(gS.pot.w)+"_"+"{:1.3e}".format(gS.pot.Ng)
    gS.U = np.load(src+"data/"+name_func+".npy")