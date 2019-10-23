/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This code computes the interpolations for the GPU PhenomHM waveform.
 *  This is implemented on the CPU to mirror the GPU program.
 *
 *
 *  Copyright (C) 2019 Michael Katz
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#ifndef __INTERPOLATE_H_
#define __INTERPOLATE_H_

#ifdef __CUDACC__
#include<cuda_runtime_api.h>
#include <cuda.h>
#endif

class Interpolate{
    double *w;
    double *a;
    double *b;
    double *c;
    double *D;
    double *x;

    double *d_b;
    double *d_c;
    double *d_w;
    double *d_x;
    int m;
    int n;
    int to_gpu;
    size_t bufferSizeInBytes;
    void *pBuffer;

    #ifdef __CUDACC__
    cudaError_t err;
    #endif

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();


    #ifdef __CUDACC__
    __host__
    #endif
    void alloc_arrays(int m, int n, double *d_B);
    #ifdef __CUDACC__
    __host__
    #endif
    void prep(double *B, int m_, int n_, int to_gpu_);

    #ifdef __CUDACC__
    __host__
    #endif
    ~Interpolate(); //destructor

    #ifdef __CUDACC__
    __host__
    #endif
    void gpu_fit_constants(double *B);
    #ifdef __CUDACC__
    __host__
    #endif
    void fit_constants(double *B);
    #ifdef __CUDACC__
    __host__
    #endif
    void cpu_fit_constants(double *B);
};

#endif //__INTERPOLATE_H_
