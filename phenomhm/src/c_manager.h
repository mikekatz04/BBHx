/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This is the central piece of code. This file implements a class
 *  that takes data in on the cpu side and mimics the GPU version.
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
#include <complex>
#include "PhenomHM.h"
#include "c_interpolate.h"

class PhenomHM {
  // pointer to the GPU memory where the array is stored
  int current_status;
  int max_length_init;
  int data_stream_length;
  double *freqs;
  int current_length;
  double m1; //solar masses
  double m2; //solar masses
  double chi1z;
  double chi2z;
  double distance;
  double phiRef;
  double deltaF;
  double fRef;
  unsigned int *l_vals;
  unsigned int *m_vals;
  int num_modes;
  int to_gpu;
  PhenomHMStorage *pHM_trans;
  IMRPhenomDAmplitudeCoefficients *pAmp_trans;
  AmpInsPrefactors *amp_prefactors_trans;
  PhenDAmpAndPhasePreComp *pDPreComp_all_trans;
  HMPhasePreComp *q_all_trans;
  double t0;
  double phi0;
  double amp0;
  int retcode;
  double m1_SI;
  double m2_SI;
  cmplx *H_mat;

  Interpolate interp;

  double *cShift;

  // Interpolate related stuff
  //Interpolate interp;


  ModeContainer *mode_vals;

  double *data_freqs;
  cmplx *data_channel1;
  cmplx *data_channel2;
  cmplx *data_channel3;

  cmplx *template_channel1;
  cmplx *template_channel2;
  cmplx *template_channel3;

  double *channel1_ASDinv;
  double *channel2_ASDinv;
  double *channel3_ASDinv;

  double inc;
  double lam;
  double beta;
  double psi;
  double t0_epoch;
  double tRef_wave_frame;
  double tRef_sampling_frame;
  int TDItag;
  double t_obs_dur;
  double merger_freq;

  double *B;


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
      double *data_freqs_,
      cmplx *data_channel1_,
      cmplx *data_channel2_,
      cmplx *data_channel3_, int data_stream_length_, double *X_ASDinv_, double *Y_ASDinv_, double *Z_ASDinv_, int TDItag,
      double t_obs_dur_); // constructor (copies to GPU)

  ~PhenomHM(); // destructor

    void gen_amp_phase(double *freqs_, int current_length_,
            double m1_, //solar masses
            double m2_, //solar masses
            double chi1z_,
            double chi2z_,
            double distance_,
            double phiRef_,
            double fRef_);

  void setup_interp_wave();

  void LISAresponseFD(double inc, double lam, double beta, double psi, double t0_epoch, double tRef_wave_frame_, double tRef_sampling_frame_, double merger_freq);

  void setup_interp_response();

  void perform_interp();

  void Likelihood (double *like_out_);

  void GetAmpPhase(double* amp_, double* phase_);

  void Combine();

  void GetTDI(cmplx *data_channel1_, cmplx *data_channel2_, cmplx *data_channel3_);
/*
  void setup_interp_wave();

  void LISAresponseFD(double inc, double lam, double beta, double psi, double t0_epoch, double tRef, double merger_freq);

  void setup_interp_response();

  void perform_interp();

  void Likelihood (double *like_out_);

  void GetTDI (cmplx* X_, cmplx* Y_, cmplx* Z_);

  void GetAmpPhase(double* amp_, double* phase_);*/
};


#endif //__MANAGER_H__
