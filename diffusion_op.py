#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The dynamical generator of the diffusion equation
"""
import numpy as np

def diff(func,tup):
    ''' Dynamical generator of the diffusion equation
        
        Input:
            func: array, the initial state 
            tup: tuple, containing the following variables
            x_max: float, max(x), where x is the grid
            x_min: float, min(x)
            lx: int, length of x
            lam_min: float, minimum eigenvalue
            lam_max: float, maximum eigenvalue
            dk: float, the spacing in k-space
            D: diffusion constant
        
        Return: Normalized version of H
    '''
    x,max_x,min_x,lx,lam_min,lam_max,dk,D = tup
    fftPfunc = np.fft.fft(func)
    

    
    kx = np.array(list(range(0,lx//2+1))+list(range(-lx//2+1,0)))*dk
    dx2fft =  ((1j*kx)**2)*fftPfunc
    dx2 = np.fft.ifft(dx2fft)                  # Second partial derivative of x
    df = D*dx2
    dE =lam_max-lam_min
    
    # The normalized operator, in the appropriat form for the Chebychev propagator.
    O = (2/dE)*(df-lam_min*func)-func
    return O

