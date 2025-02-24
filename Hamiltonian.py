#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonic oscillator Hamiltonian for the Chebychev propagation scheme.

The funciton Hamiltonian implements the Harmonic oscillator Hamiltonian utilizing 
FFT. The differential operator is then normalized and shifted inorder to fit for 
the Chebychev propagation scheme.
"""
import numpy as np

def Hamiltonian(func,x,max_x,min_x,lx,m,k,lam_min,lam_max,dk):
    ''' Normalized Hamiltonian of a quantum harmonic oscillator
        
        Input:
            func: array, the initial state 
            x_max: float, max(x), where x is the grid
            x_min: float, min(x)
            lx: int, length of x
            m: float, mass
            k: float, spring constant
            lam_min: float, minimum eigenvalue
            lam_max: float, maximum eigenvalue
            dk: float, the spacing in k-space
        
        Return: Normalized version of H
    '''
    fftPfunc = np.fft.fft(func)
    

    
    kx = np.array(list(range(0,lx//2+1))+list(range(-lx//2+1,0)))*dk
    dx2fft =  ((1j*kx)**2)*fftPfunc;
    dx2 = np.fft.ifft(dx2fft)                  # Second partial derivative of x
    df = -0.5*(1/m)*dx2+0.5*k*(x**2)*func;
    dE =lam_max-lam_min;
    
    # The normalized operator, in the appropriat form for the Chebychev propagator.
    O = (2/dE)*(df-lam_min*func)-func
    return O

