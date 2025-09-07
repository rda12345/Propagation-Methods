#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluates the coefficients of the Chebychev series of exp(-i*O*dt/hbar),
where O is a linear differential operator and i = sqrt(-1) 
Based on the paper
R. Kosloff, Propagation Methods for Quantum Molecular Dynamics,
Annu. Rev. Phys. Chem., 45, 145-178 (1994)
"""
import pylab,scipy,numpy


def besselj(v,z):
    '''
    Bessel function of the first kind of real order and complex argument.
    v: float real, order of the function
    z: float complex, argument   
    '''
    return scipy.special.jv(v,z)

def cheby_coeff(dt,hbar,lam_max,lam_min):
    '''
    Evaluates the Chebychev coefficients for the propagator exp(-i*O*dt/hbar),
    where O is a linear differential operator and i is the imaginary number. 
    
    Based on the paper: Propagation Methods for Quantum Molecular Dynamics - Ronnie Kosloff 1994
    
    dt: float, the time interval of propagation.
    hbar: float, Planck's reduced constant
    lam_min: float, expected minimum eigenvalue of O
    lam_max: float, expected max eigenvalue of O
    
    Note: lam_min and lam_max do not have to be exact. Although it is important 
          to keep the normalized range of eigenvalues within [-1,1].
    
    Retun: array, contains the Chebychev coefficients, which can be used to propagate the system.
    
    '''
    # Factor to ajust the eigenvalues to be between [-1,1]
    dE = lam_max-lam_min
    R = dE*dt/(hbar*2)
    Nmax = 100000        # Maximum number of coefficients 
    c_j = numpy.zeros(Nmax,dtype = complex)
    c_j[0] = besselj(0,R)           #Zero coefficient.
    # The vector a contans c_n coefficients from 1 to Nmax, further multipication by exp(lam_min+ *dt) is needed  
    for n in range(1,Nmax):
        c_j[n] = (2*1j**n)*besselj(n,R)
        if abs(c_j[n])<1e-17 and n>R: 
            break

    Nmax = n+1
    d_j = numpy.exp((1j/hbar)*(dE/2+lam_min)*dt)*c_j #normalizing factor times the cooefficeints
    return d_j,Nmax


## Test 

# dt = 0.1
# hbar = 1
# lam_max = 10
# lam_min = -10
# d_j,Nmax = cheby_coeff(dt,hbar,lam_max,lam_min)
# print('d_j: ', d_j)
# print('Nmax: ', Nmax)

