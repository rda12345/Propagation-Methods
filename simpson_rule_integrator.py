#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The file contains an integrator of 1D functions using Simpson's rule.
"""
import numpy as np
import math

pi = math.pi
def  Simpson_rule_integrator(x,f):
    '''
    Simpson rule integrator of f(x)
    NOTE: N should be odd
    
    Input
        x: pylab array, sample of points in the integration range
        f: pylab array, the function to be integrated
        
    Returns: 
        result: float, the integration result
        errorscale: float, the error.
    '''
    N = len(x)
    h = (x[-1]-x[0])/(N-1)
    result = (h/3)*(f[0] + 2*sum(f[range(2,N-2,2)])+ 4*sum(f[range(1,N-1,2)])+f[-1])
    error_scale = h**4
    return result,error_scale


# Test

dx = 0.01
sig = 1
x = np.arange(-8,8,dx)
y = np.exp(-x**2/(2*sig**2))

res = Simpson_rule_integrator(x, y)
print('real error: ', math.sqrt(2*pi*sig**2)-res[0])
print('theoretical error: ', res[1])
