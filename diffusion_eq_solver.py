#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion equation solver
"""
import numpy as np
from diffusion_op import diff
from cheby_coeff_real import cheby_coeff, propagator
import matplotlib.pyplot as plt

# Constants
D = 2       # diffusion constant
A = 1       # initial state amplitude
sig = 2     # initial state standard deviation
pi = np.pi


# Model constants paramenter definition
L = 20        # length of the placment grid
Nx = 2**10              # number of grid points      
dx = L/Nx

#x = np.transpose(np.array(range(-L/2,dx,L/2)))
x = np.arange(-L/2,L/2,dx)
kmax = pi/dx
dk = 2*pi/(Nx*dx)


# Initial wave function
u0 = A*np.exp(-x**2/(2*sig**2))


# Plot of initial wavefunction
#plt.figure(1)
#plt.plot(x,u0)
#plt.show()

## Chebychev coefficents decleration
dt = 0.0001
lam_max = D*kmax**2
lam_min = 0 
R = (lam_max-lam_min)*dt/2
d_j,Nmax = cheby_coeff(dt,lam_max,lam_min)  

## Declarations for the time propagation
lx = len(x)
max_x = max(x)
min_x = min(x)  

## Propagation in time
Nt = 10**4
tup = (x,max_x,min_x,lx,lam_min,lam_max,dk,D)
u_t = propagator(u0,d_j,Nmax,diff,Nt,tup)

## Analytical solution
t = Nt*dt
sig_t = np.sqrt(sig**2 + 2*D*t)
u_analytic = A*(sig/sig_t)*np.exp(-x**2/(2*sig_t**2))


## Plots
plt.figure(2)
plt.plot(x,u_t,'r',label = 'numerical')
plt.plot(x,u_analytic,'-.k',label = 'analytical')
plt.legend()
plt.show()
    

