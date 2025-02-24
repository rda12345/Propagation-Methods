#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the momoentum operator using the descrite fourier transform.
"""
import numpy as np
import math

def build_momentum(N,dx):
    ''' 
    Builds the momentum operator using the descrite fourier transform.
    
    N: int, the size of the grid
    dx: float, the grid interval
    '''
    pi = math.pi
    dk = 2*pi/(N*dx)

    
    F_DFT = np.zeros((N,N),dtype = complex)
    B_DFT = np.zeros((N,N),dtype = complex)
    P_DFT = np.zeros((N,N),dtype = complex)
    w = np.exp(2*pi*1j/N)
    for i in range(N):
        F_DFT[i,i] = w**(-(i**2))
        B_DFT[i,i] = w**(i**2)
        if i <= N/2:
            P_DFT[i,i] = i*dk
            
        else:
            P_DFT[i,i] = (-N+i)*dk    
        for j in range(i):
            F_DFT[i,j] = w**(-i*j)
            F_DFT[j,i] = F_DFT[i,j]
            B_DFT[i,j] = w**(i*j)
            B_DFT[j,i] = B_DFT[i,j]
    #print(P_DFT)  
    return (1/N)*B_DFT@P_DFT@F_DFT



# Test 
# N = 4
# dx = 0.1
# print(build_momentum(N, dx))