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

class Interpolate{
    double *w;
    double *d_mat;

    double *dl;
    double *d;
    double *du;
    int m;
    int n;
    int to_gpu;

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();

    void prep(double *b_mat, int m_, int n_, int to_gpu_);

    ~Interpolate(); //destructor

    void fit_constants(double *b_mat);
};

void host_fill_B_wave(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_fill_B_response(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_set_spline_constants_wave(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_set_spline_constants_response(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_interpolate(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals, int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv, double t_obs_dur);

#endif //__INTERPOLATE_H_
