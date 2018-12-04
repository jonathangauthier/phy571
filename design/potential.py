import numpy as np
import matplotlib.pyplot as plt

class potential:
    """Specific class to define potential and its parameters.
    double_well : creates a potential with two distinct "holes"
    """
    
    def __init__(self,wx,wy,Ng,m=1,hbar=1,double_well=False):
        self.wx = wx
        self.wy = wy
        self.Ng = Ng
        self.m = m
        self.hbar = hbar
        self.double_well = double_well
        
        def V(x,y):
            """Harmonic potential of the BEC
            For the double well, a gaussian in x is added to V,
            it requires wx and wy to be nearly equal"""
            if self.double_well == False:
                return 0.5*self.m*((self.wx*x)**2+(self.wy*y)**2)
            else:
                return 0.5*self.m*((self.wx*x)**2+(self.wy*y)**2\
                        +300*self.wx**2*np.exp(-(x/4)**2/2))
            
        def Veff(x,y,u):
            """Effective potential of the BEC"""
            return V(x,y)+ self.Ng*np.abs(u)**2
            
        def f(x,y,u):
            """Function of Veff used for the computations"""
            return 1/(1j*self.hbar)*Veff(x,y,u)*u
          
        self.V = V
        self.Veff = Veff    
        self.f = f
        
    def plot_V(self,l=20,resol=4):
        """Plots the potential"""
        n = resol*l*2+1
        mat = [[self.V(x/resol-l,y/resol-l)for x in range(n)]for y in range(n)]
        plt.imshow(mat)
        plt.show()
    
    def remove_barrier(self):
        """Removes the potential barrier between the two walls"""
        self.double_well = False

"""
      
pot.plot_V()
pot.remove_barrier()      
pot.plot_V()
"""
