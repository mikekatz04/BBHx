from scipy.interpolate import CubicSpline as spl
import numpy as np

amps, freqs = np.load('spline_test.npy')

spline = spl(freqs, amps)

x = freqs
y = amps
dx = np.diff(x)

n = len(y)

dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
slope = np.diff(y, axis=0) / dxr

# Find derivative values at each x[i] by solving a tridiagonal
# system.
A = np.zeros((3, n))  # This is a banded matrix representation.
b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

# Filling the system for i=1..n-2
#                         (x[i-1] - x[i]) * s[i-1] +\
# 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
#                         (x[i] - x[i-1]) * s[i+1] =\
#       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
#           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
A[0, 2:] = dx[:-1]                   # The upper diagonal
A[-1, :-2] = dx[1:]                  # The lower diagonal

b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

A[1, 0] = dx[1]
A[0, 1] = x[2] - x[0]
d = x[2] - x[0]
b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] + dxr[0]**2 * slope[1]) / d

A[1, -1] = dx[-2]
A[-1, -2] = x[-1] - x[-3]
d = x[-1] - x[-3]
b[-1] = ((dxr[-1]**2*slope[-2] + (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)


for i in range(1, n): 
    ind_i = (param * n + i) * nsub + sub_i;
    ind_im1 = (param * n + (i-1)) * nsub + sub_i;


    w = a[ind_i]/b[ind_im1];
    b[i] = b[ind_i] - w * c[ind_im1];
    d[i] = d[ind_i] - w * d[ind_im1];


ind_i = (param * n + (n-1)) * nsub + sub_i;

d[ind_i] = d[ind_i]/b[ind_i];
for (int i = n - 2; i >= 0; i -= 1):
    ind_i = (param * n + i) * nsub + sub_i;
    ind_ip1 = (param * n + (i+1)) * nsub + sub_i;

    d[ind_i] = (d[ind_i] - c[ind_i] * d[ind_ip1])/b[ind_i];


breakpoint()


print('fin')