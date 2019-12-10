/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This code computes the fast Fourier domain response function for LISA
 *  based on Marsat and Baker 2018. This code contains the CPU side to mirror GPU functions
 *  for this calculation.
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

#ifndef _FD_RESPONSE_
#define _FD_RESPONSE_

#include <complex>
#include "globalPhenomHM.h"

typedef cmplx cmplx;

typedef struct tagGslr_holder{
    cmplx G21;
    cmplx G12;
    cmplx G23;
    cmplx G32;
    cmplx G31;
    cmplx G13;
} Gslr_holder;

typedef struct tagtransferL_holder{
    cmplx transferL1;
    cmplx transferL2;
    cmplx transferL3;
    double phaseRdelay;
} transferL_holder;

double sinc(double x);
cmplx SpinWeightedSphericalHarmonic(int s, int l, int m, double theta, double phi);
void dot_product_2d(double out[3][3], double arr1[3][3], int m1, int n1, double arr2[3][3], int m2, int n2);

double dot_product_1d(double arr1[3], double arr2[3]);

cmplx vec_H_vec_product(double arr1[3], cmplx *H, double arr2[3]);

void prep_H_info(cmplx *H_mat, unsigned int *l_vals, unsigned int *m_vals, int num_modes, double inc, double lam, double beta, double psi, double phi0);

Gslr_holder EvaluateGslr(double t, double f, cmplx *H, double k[3], int response);

transferL_holder TDICombinationFD(Gslr_holder Gslr, double f, int TDItag, int rescaled);

transferL_holder JustLISAFDresponseTDI(cmplx *H, double f, double t, double lam, double beta, double t0, int TDItag, int order_fresnel_stencil);

void JustLISAFDresponseTDI_wrap(ModeContainer *mode_vals, cmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double inc, double lam, double beta, double psi, double phi0, double t0, double tRef_wave_frame, double tRef_sampling_frame, double merger_freq, int TDItag, int order_fresnel_stencil);



#endif // _FD_RESPONSE_
