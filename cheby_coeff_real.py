#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates the coefficients of the Chebychev series of exp(O*dt),
where O is a linear differential operator.
Based on the paper
R. Kosloff, Propagation Methods for Quantum Molecular Dynamics,
Annu. Rev. Phys. Chem., 45, 145-178 (1994)
"""

## Defining axilary functions
import numpy as np
import scipy

def besseli(v,z):
    '''
    Modified Bessel function of the first kind of real order.
    Input:
        v: float real, order of the function
        z: float or complex, argument   
    '''
    return scipy.special.iv(v,z)

def cheby_coeff(dt,lam_max,lam_min):
    '''
    Evaluates the Chebychev coefficients for the propagator exp(O*dt),
    where O is a linear differential operator with units of 1/time.
    
    Based on the paper: Propagation Methods for Quantum Molecular Dynamics - Ronnie Kosloff 1994

    Input:
        dt: float, the time interval of propagation.
        lam_min: float, expected minimum eigenvalue of O
        lam_max: float, expected max eigenvalue of O
    
    Note: lam_min and lam_max do not have to be exact. Although it is important 
          to keep the normalized range of eigenvalues within [-1,1].
    
    Retun: array, contains the Chebychev coefficients, which can be used to propagate the system.
    
    '''
    # Factor to ajust the eigenvalues to be between [-1,1]
    dE = lam_max-lam_min  #Note that dE has units of 1/time
    R = dE*dt/2
    Nmax = 1000        # Maximum number of coefficients 
    c_j = np.zeros(Nmax,dtype = complex)
    c_j[0] = besseli(0,R)           #Zero coefficient.
    # The vector a contans c_n coefficients from 1 to Nmax, further multipication by exp(lam_min+ *dt) is needed  
    for n in range(1,Nmax):
        c_j[n] = 2*besseli(n,R)
        if abs(c_j[n])<1e-17 and n>R: 
            break

    Nmax = n+1
    d_j = np.exp((lam_min+lam_max)*dt/2)*c_j #normalizing factor times the cooefficeints
    return d_j,Nmax


def propagator(vec,d_j,Nmax,O,Nt,tup):
    """
    Evaluates exp(O*t)*vec, with t=Nt*dt, utilizing a Chebychev expansion of the dynamical propagator.

    Input: 
        vec: numpy array, containing the initial state
        dj: numpy array, the Chebychev coefficients
        O: function, the dynamical generator (e.g, a differential operator)
        tup: tuple, containing the dynamical generator additional parameters
    """
    fi = np.zeros((vec.shape[0],3),dtype = complex)
    #x,max_x,min_x,lx,mass,k,lam_min,lam_max,dk = tup
    for j in range(Nt):     # Running over the time steps
        fi[:,0] = vec
        # The normalized differential operator O
        fi[:,1] = O(vec,tup)              
        G1 = d_j[0]*fi[:,0]
        G2 = d_j[1]*fi[:,1]
        G_3 = G1+G2
        for i in range(1,Nmax-1):
            fi[:,2] = 2*O(fi[:,1],tup)-fi[:,0]
            fi[:,0]=fi[:,1]
            fi[:,1]=fi[:,2]
            G_3 = G_3+d_j[i+1]*fi[:,2]
        #norm[j] = sum((np.abs(G_3))**2)*dx;
        vec = G_3
    return vec

######################### Test 1 - evaluating exp(dt) #########################
#vec = np.array([2])
#Nt = 1
#def O(vec,tup):
    #return vec 
#dt = 10
#tup = 0
#lam_max, lam_min = 1, -1
#d_j, Nmax = cheby_coeff(dt,lam_max, lam_min)
#ans = propagator(vec,d_j,Nmax,O,Nt,tup)
#print(f'numerical result = {ans}')
#print(f'analtycial result = {2*np.exp(dt)}')
###############################################################################

    








