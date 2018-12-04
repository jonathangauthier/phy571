import numpy as np
import os

def get_ground(gS, threshold=1e-8):
    gS.renorm()
    gS.step()
    while np.sum((np.abs(gS.oldU-gS.U))**2)>threshold:
        gS.renorm()
        gS.step()
        gS.renorm()
    return gS.U
    
    
def save_ground(gS,src,overwrite=False):
    name_func = "grd_state_2D_"+\
                str(int(gS.xMax))+"_"+\
                "{:1.1e}".format(gS.dtau)+"_"+\
                str(int(gS.Jx))+"_"+\
                "{:1.1e}".format(gS.pot.wx)+"_"+\
                "{:1.1e}".format(gS.pot.Ng)
                
    if overwrite or not os.path.isfile(src+"data\\"+name_func):
        grd_wave_func = get_ground(gS)
        np.savetxt(src+"data\\"+name_func+".txt",grd_wave_func)


def load_ground(gS,src):
    name_func = "grd_state_2D_"+\
                str(int(gS.xMax))+"_"+\
                "{:1.1e}".format(gS.dtau)+"_"+\
                str(int(gS.Jx))+"_"+\
                "{:1.1e}".format(gS.pot.wx)+"_"+\
                "{:1.1e}".format(gS.pot.Ng)
    return np.loadtxt(src+"data\\"+name_func+".txt",dtype=complex)
    