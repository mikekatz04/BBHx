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
#include <cusparse_v2.h>

class Interpolate{
    double *w;
    double *D;

    double *d_b;
    double *d_c;
    double *d_w;
    double *d_x;
    cusparseHandle_t  handle;
    cudaError_t err;
    int m;
    int n;
    int to_gpu;
    size_t bufferSizeInBytes;
    void *pBuffer;

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();

    __host__ void alloc_arrays(int m, int n, double *d_B);
    __host__ void prep(double *B, int m_, int n_, int to_gpu_);

    __host__ ~Interpolate(); //destructor

    __host__ void gpu_fit_constants(double *B);
    __host__ void fit_constants(double *B);
};

#endif //__INTERPOLATE_H_
