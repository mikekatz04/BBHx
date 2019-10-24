/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This is the central piece of code. This file implements a class
 *  that takes data in on the cpu side, copies
 *  it to the gpu, and exposes functions that let
 *  you perform actions with the GPU.
 *
 *  This class will get translated into python via cython.
 *
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

#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "globalPhenomHM.h"
#include "PhenomHM.h"
#include "interpolate.hh"

#ifdef __CUDACC__
#include<cuda_runtime_api.h>
#include <cuda.h>
#include "cublas_v2.h"
#endif

class PhenomHM {
  // pointer to the GPU memory where the array is stored
  int current_status;
  int max_length_init;
  int data_stream_length;
  double *freqs;
  int current_length;
  int nwalkers;
  double* m1; //solar masses
  double* m2; //solar masses
  double* chi1z;
  double* chi2z;
  double* distance;
  double* phiRef;
  double* f_ref;
  double* cShift;
  unsigned int *l_vals;
  unsigned int *m_vals;
  int num_modes;
  int to_gpu;
  PhenomHMStorage *pHM_trans;
  IMRPhenomDAmplitudeCoefficients *pAmp_trans;
  AmpInsPrefactors *amp_prefactors_trans;
  PhenDAmpAndPhasePreComp *pDPreComp_all_trans;
  HMPhasePreComp *q_all_trans;
  double* t0;
  double* phi0;
  double* amp0;
  int retcode;
  cmplx *H;
  double* M_tot_sec;

  double **d_freqs;
  double **d_time_freq_corr;
  unsigned int *d_l_vals;
  unsigned int *d_m_vals;
  PhenomHMStorage **d_pHM_trans;
  IMRPhenomDAmplitudeCoefficients **d_pAmp_trans;
  AmpInsPrefactors **d_amp_prefactors_trans;
  PhenDAmpAndPhasePreComp **d_pDPreComp_all_trans;
  HMPhasePreComp **d_q_all_trans;
  double **d_cShift;
  cmplx **d_H;
  double** d_t0;
  double** d_phi0;
  double** d_amp0;
  double** d_M_tot_sec;

  int NUM_THREADS;
  int num_blocks;

  #ifdef __CUDACC__
  cublasHandle_t *handle;
  cublasStatus_t stat;
  #endif

  // Interpolate related stuff
  Interpolate *interp;
  double **d_B;
  double *B;

  ModeContainer *mode_vals;
  ModeContainer **d_mode_vals;
  int ndevices;

  cmplx * template_channel1;
  cmplx * template_channel2;
  cmplx * template_channel3;

  double *data_freqs;
  cmplx *data_channel1;
  cmplx *data_channel2;
  cmplx *data_channel3;

  double *channel1_ASDinv;
  double *channel2_ASDinv;
  double *channel3_ASDinv;

    double **d_data_freqs;
  cmplx **d_data_channel1;
  cmplx **d_data_channel2;
  cmplx **d_data_channel3;

  cmplx **d_template_channel1;
  cmplx **d_template_channel2;
  cmplx **d_template_channel3;
  double **d_channel1_ASDinv;
  double **d_channel2_ASDinv;
  double **d_channel3_ASDinv;

  double* inc;
  double* lam;
  double* beta;
  double* psi;
  double* t0_epoch;
  double* tRef_wave_frame;
  double* tRef_sampling_frame;
  int TDItag;
  double t_obs_dur;
  double* merger_freq;

  double** d_inc;
  double** d_lam;
  double** d_beta;
  double** d_psi;
  double** d_t0_epoch;
  double** d_tRef_wave_frame;
  double** d_tRef_sampling_frame;
  double** d_merger_freq;
  double** d_phiRef;
  double d_log10f;


public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  PhenomHM(int max_length_init_,
      unsigned int *l_vals_,
      unsigned int *m_vals_,
      int num_modes_,
      int data_stream_length_,
      int TDItag,
      double t_obs_dur_,
      int nwalkers_,
      int ndevices_); // constructor (copies to GPU)

  ~PhenomHM(); // destructor

  void input_data(double *data_freqs, double *data_channel1,
                            double *data_channel2, double *data_channel3,
                            double *channel1_ASDinv, double *channel2_ASDinv,
                            double *channel3_ASDinv, int data_stream_length_);

    void gen_amp_phase(double *freqs_, int current_length_,
        double* m1_, //solar masses
        double* m2_, //solar masses
        double* chi1z_,
        double* chi2z_,
        double* distance_,
        double* phiRef_,
        double* f_ref_);

    void gen_amp_phase_prep(int ind_walker_, double *freqs_, int current_length_,
            double m1_, //solar masses
            double m2_, //solar masses
            double chi1z_,
            double chi2z_,
            double distance_,
            double phiRef_,
            double f_ref_);

  void setup_interp_wave();

  void LISAresponseFD(double* inc_, double* lam_, double* beta_, double* psi_, double* t0_epoch_, double* tRef_wave_frame_, double* tRef_sampling_frame_, double* merger_freq_);

  void setup_interp_response();

  void perform_interp();

  void Likelihood (double *d_h_arr, double *h_h_arr);

  void GetTDI (double* X_, double* Y_, double* Z_);

  void GetAmpPhase(double* amp_, double* phase_);
};


#endif //__MANAGER_H__
