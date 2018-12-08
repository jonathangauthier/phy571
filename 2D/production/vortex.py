import numpy as np

class vortex:
    """Vortex centered in (xv,yv) with  a typical width xi and an integer-valued winding number p"""
    
    def __init__(self,gS,xv,yv,xi,p):
        self.xv = xv
        self.yv = yv
        self.xi = xi
        self.p =p
        
        self.rho = np.sqrt((gS.X-self.xv)**2 + (gS.Y-self.yv)**2)
        self.rho_n = self.rho/self.xi
        self.theta = np.arctan2((gS.Y-self.yv),(gS.X-self.xv))

def shape_vortex(gS,vtx):
    """create a hole in the BEC gS centered in (xv,yv) with a typical width xi"""
    gS.U = gS.U * vtx.rho_n/np.sqrt(2+vtx.rho_n**2)


def force_phase(gS,vtx):
    """create a vortex in the BEC gS centered in (xv,yv) with an integer-valued winding number p"""
    gS.U = np.abs(gS.U) * np.exp(1j*vtx.p*vtx.theta)