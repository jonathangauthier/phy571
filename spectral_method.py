#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:40:42 2018

@author: jonathan.gauthier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


class Spectral:
    '''Spectral stepping method for solving PDEs'''
    def __init__(self, L, number_of_points, D, V):
        '''Initialises'''
        self.L = L
        self.number_of_points = number_of_points
        self.D = D
        self.time = []
        self.k = 2 * np.pi * np.fft.fftfreq(self.number_of_points)
        self.space, self.dx = np.linspace(-self.L/2, self.L/2,self.number_of_points, retstep = True)
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
        ufft *= np.exp((- self.D * self.k**2)* dt * (-1j))
        #print(ufft)
        unew = np.fft.ifft(ufft)
        #unew[0] = 0
        #unew[-1] = 0
        unew *= np.exp(-1j * self.V(self.space) * dt)
        #print((np.sum(np.abs(unew)**2*self.dx))**0.5)
        unew /= (np.sum(np.abs(unew)**2*self.dx))**0.5
        self.u.append(unew)
        return
   
    def plot_last_step(self):
        '''Plots the last step computed'''
        plt.plot(self.space, np.abs(self.u[-1])**2)
        return


def advance(spectral, T):
    """Advances spectral of time T"""
    dt = spectral.dx**2
    for i in range(int(T/dt)):
        spectral.step((1j)*dt) #The factor should be dt for normal evolution and 1j for imaginary time evolution
    return


def V(x):
    return 0.5 * x**2


gauss = Gaussian(1,1,0)

spectral = Spectral(20, 1001, 0.5, V)
spectral.initialize(gauss)

fig,ax = plt.subplots()
line, = plt.plot([],[]) 
plt.xlim(-spectral.L/2, spectral.L/2)
plt.ylim(-0.1,3)
count = 0

def init():
    line.set_data([],[])
    return line,


def animate(i):
    advance(spectral, 1)
    line.set_data(spectral.space, np.abs(spectral.u[-1])**2)
    ax.set_title("{:1}".format(count))
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, blit=True, interval=10, repeat=False)

plt.show()

print("done")

np.savetxt("groundstate.txt",spectral.u)
