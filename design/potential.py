import numpy as np

class potential:
    """Specific class to define potential and its parameters"""
    
    def __init__(self,wx,wy,Ng,m=1,hbar=1):
        self.wx = wx
        self.wy = wy
        self.Ng = Ng
        self.m = m
        self.hbar = hbar
        
        def V(x,y):
            """Potential of the BEC"""
            return 0.5*self.m*((self.wx*x)**2+(self.wy*y)**2)
            
        def Veff(x,y,u):
            """Effective potential of the BEC"""
            return V(x,y)+ self.Ng*np.abs(u)**2
            
        def f(x,y,u):
            return 1/(1j*self.hbar)*Veff(x,y,u)*u
          
        self.V = V
        self.Veff = Veff    
        self.f = f