import numpy as np
import matplotlib.pyplot as plt
import math

class Gaussian:
    """Gaussian function"""
    def __init__(self, width, amplitude, x0):
        """Create Gaussian with the given width and amplitude, centered at x = x0"""
        self.width = width
        self.amplitude = amplitude
        self.x0 = x0
       

    def __call__(self, x):
        return math.exp(-0.5 * ((x-self.x0)/self.width)**2)

#gauss = Gaussian(1,1,0.5)

class Spectral:
    '''Spectral stepping method for solving PDEs'''
    def __init__(self, L, number_of_points, D, V):
        '''Initialises'''
        self.L = L
        self.number_of_points = number_of_points
        self.D = D
        self.time = []
        self.k = 2 * np.pi * np.fft.fftfreq(self.number_of_points)
        self.space = np.linspace(0,self.L,self.number_of_points)
        self.u = []
        self.V = V
        return

    def initialize(self, function):
        '''Sets the initial conditions'''
        self.time.append(0)
        self.u = [[function(x) for x in self.space]]
        return

    def step(self, dt):
        '''Make a step dt forward in time'''
        ufft = np.fft.fft(self.u[-1])
        self.time.append(self.time[-1]+dt)
        ufft *= np.exp(-(self.D * self.k**2 + 1j * self.k * self.V) * dt)
        self.u.append(np.fft.ifft(ufft))
        return
   
    def plot_last_step(self):
        '''Plots the last step computed'''
        plt.plot(self.space, self.u[-1])
        return


def advance(spectral, T, N):
    dt = T/N
    for i in range(N):
        spectral.step(dt)
    return



gauss = Gaussian(1,1,10)

spectral = Spectral(20, 1000, 1, 0)
spectral.initialize(gauss)

spectral.plot_last_step()

advance(spectral, 3000, 100)

spectral.plot_last_step()
plt.show()
