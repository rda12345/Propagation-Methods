#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-dependent Schrodinger equation for an harmonic
oscillator system. T
The Hamiltonian is: H = p^2/2m +(1/2)*m*omega^2

Method:
Using a different operator and the original forier grid kinetic operator
Taken from the article "Grid method for cold molecules: Determination of
photoassociation line shapes and rate constants
"""
import math
import numpy as np
import pylab
from build_momentum import build_momentum
import numpy.linalg as LA
from cheby_coeff import cheby_coeff
from Hamiltonian import Hamiltonian
# Constants
mass = 1
omega = 1
k = mass*omega**2
hbar = 1
pi = math.pi


# Model constants paramenter definition
L = 20        # length of the placment grid
Nx = 2**10              # number of grid points      
dx = L/Nx

#x = np.transpose(np.array(range(-L/2,dx,L/2)))
x = np.arange(-L/2,L/2,dx)
kmax = pi/dx
dk = 2*pi/(Nx*dx)

# Hamiltonian builder
P_op = build_momentum(Nx,dx)
T_mat = (P_op@P_op)/(2*mass)     # Kinetic energy
V = 0.5*k*(x**2)              # Potential energy
V_mat = np.diag(V)
H_mat = T_mat+V_mat;


eigenValues,eigenVectors = LA.eig(H_mat)
ind = eigenValues.argsort()
eigenValues = eigenValues[ind]
eigenVectors = eigenVectors[:,ind]




# Initial wave function
Psi = eigenVectors[:,0]
norm = math.sqrt(sum((np.abs(Psi)**2))*dx)
Psi=Psi/norm

# Plot of initial wavefunction
pylab.figure(1)
pylab.plot(x,np.real(Psi),'p')
pylab.plot(x,np.imag(Psi),'k')
pylab.legend()
pylab.show()

## Chebychev coefficents decleration
dt = pi
V_max = max(V)
V_min = min(V)
lam_max = (1/2*mass)*kmax**2+V_max
lam_min = V_min 
R = (lam_max-lam_min)*dt/2
d_j,Nmax = cheby_coeff(dt,hbar,lam_max,lam_min)  

## Declarations for the time propagation
lx = len(x)
max_x = max(x)
min_x = min(x)  

## Propagation in time

# Building fi the column n belongs to fi_n-1
fi = np.zeros((len(x),3),dtype = complex)
t = 1       # Number of time-steps
norm = np.zeros(t+1)
norm[0] = sum(np.abs(Psi)**2)*dx
for j in range(1,t+1):     # Running over the time steps
    fi[:,0] = Psi
    # The normalized differential operator H
    fi[:,1] = Hamiltonian( Psi,x,max_x,min_x,lx,mass,k,lam_min,lam_max,dk)              
    G1 = d_j[0]*fi[:,0]
    #print(G1)
    G2 = d_j[1]*fi[:,1]
    #print(G2)
    G_3 = G1+G2
    for i in range(1,Nmax):
        fi[:,2] = 2*Hamiltonian(fi[:,1],x,max_x,min_x,lx,mass,k,lam_min,lam_max,dk)-fi[:,0]
        fi[:,0]=fi[:,1]
        fi[:,1]=fi[:,2]
        G_3 = G_3+d_j[i+1]*fi[:,2]
    norm[j] = sum((np.abs(G_3))**2)*dx;
    Psi = G_3
pylab.figure(2)
pylab.plot(x,np.real(Psi),'p')
pylab.plot(x,np.imag(Psi),'k')
pylab.show()
    
