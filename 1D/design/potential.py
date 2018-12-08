import numpy as np

class potential:
    """Specific class to define potential and its parameters"""
    
    def __init__(self,w,Ng,m=1,hbar=1):
        self.w = w
        self.Ng = Ng
        self.m = 1
        self.hbar = 1
        
        def V(x):
            """Potential of the BEC"""
            return 0.5*self.m*(self.w*x)**2
            
        def Veff(x,u):
            """Effective potential of the BEC"""
            return V(x)+ self.Ng*np.abs(u)**2
            
        def f(x,u):
            return 1/(1j*self.hbar)*Veff(x,u)*u
          
        self.V = V
        self.Veff = Veff    
        self.f = f