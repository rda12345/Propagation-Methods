# Propagation-Methods
The repository contains an example of a linear propagation methods.
The methods are illustrated by the propagation of a 1D quantum harmonic oscillator in Schrod_eq_solver_HO.py.

It contains:
1. 1D and 2D fft based differentiation methods
2. 'Real' time Chebychev propagation method, i.e. v(t) = exp(-i*O*t)*v(0).
3. 'Imaginary' time Chebychev propagation method, i.e. v(t) = exp(O*t)*v(0).


Here, O is a a differential linear operator, v(0) is the vector at initial time and v(t) is a vector at time t.

