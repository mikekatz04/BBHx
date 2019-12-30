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

#ifdef __CUDACC__

#include "cuComplex.h"
#include "cublas_v2.h"
#include <kernel.cu>
#include <likelihood.cu>
#include "createGPUHolders.cu"
#include "kernel_response.cu"
#include "interpolate.cu"
#include <cuda_runtime_api.h>
#include <cuda.h>

#else

#include <kernel.cpp>
#include <likelihood.cpp>
#include "kernel_response.cpp"
#include "interpolate.cpp"

#endif

#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "fdresponse.h"

#include "omp.h"
#include "cuda_complex.hpp"
// TODO: CUTOFF PHASE WHEN IT STARTS TO GO BACK UP!!!

using namespace std;

#ifdef __CUDACC__
void print_mem_info(){
        // show memory usage of GPU

        cudaError_t cuda_status;

        size_t free_byte ;

        size_t total_byte ;

        cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

        if ( cudaSuccess != cuda_status ){

            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

            exit(1);

        }



        double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

#endif

PhenomHM::PhenomHM (int max_length_init_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    double *data_freqs_,
    cmplx *data_channel1_,
    cmplx *data_channel2_,
    cmplx *data_channel3_, int data_stream_length_,
    double *channel1_ASDinv_, double *channel2_ASDinv_, double *channel3_ASDinv_,
    int TDItag_,
    double t_obs_start_,
    double t_obs_end_,
    int nwalkers_,
    int ndevices_){

    #pragma omp parallel
    {
      if (omp_get_thread_num() == 1) printf("NUM OMP THREADS: %d\n", omp_get_num_threads());
    }


    max_length_init = max_length_init_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    data_freqs = data_freqs_;
    data_stream_length = data_stream_length_;
    channel1_ASDinv = channel1_ASDinv_;
    channel2_ASDinv = channel2_ASDinv_;
    channel3_ASDinv = channel3_ASDinv_;
    data_channel1 = data_channel1_;
    data_channel2 = data_channel2_;
    data_channel3 = data_channel3_;
    nwalkers = nwalkers_;

    TDItag = TDItag_;
    t_obs_start = t_obs_start_;
    t_obs_end = t_obs_end_;

    #ifdef __CUDACC__
    to_gpu = 1;

    #else
    to_gpu = 0;
    assert(ndevices == 1);
    #endif

    ndevices = ndevices_;


    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage[nwalkers*ndevices];

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients[nwalkers*ndevices];

    amp_prefactors_trans = new AmpInsPrefactors[nwalkers*ndevices];

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes*nwalkers*ndevices];

    q_all_trans = new HMPhasePreComp[num_modes*nwalkers*ndevices];

    t0 = new double[nwalkers*ndevices];

    phi0 = new double[nwalkers*ndevices];

    amp0 = new double[nwalkers*ndevices];

    // malloc and setup for the GPU

  mode_vals = cpu_create_modes(num_modes, nwalkers*ndevices, l_vals, m_vals, max_length_init, to_gpu, 1);

  // phase shifts for each m mode
  cShift = new double[7];
  cShift[0] = 0.0;
  cShift[1] = PI_2,
  cShift[2] = 0.0;
  cShift[3] = -PI_2;
  cShift[4] = PI;
  cShift[5] = PI_2;
  cShift[6] = 0.0;

  //double cShift[7] = {0.0,
  //                     PI_2 /* i shift */,
  //                     0.0,
  //                     -PI_2 /* -i shift */,
  //                     PI /* 1 shift */,
  //                     PI_2 /* -1 shift */,
  //                     0.0};

   H = new cmplx[9*nwalkers*num_modes*ndevices];

   M_tot_sec = new double[nwalkers*ndevices];

   interp = new Interpolate[ndevices];

   #ifdef __CUDACC__

  cudaError_t err;

  d_mode_vals = new ModeContainer*[ndevices];
  d_freqs = new double*[ndevices];
  d_H = new agcmplx*[ndevices];
  d_B = new double*[ndevices];

  d_template_channel1 = new agcmplx*[ndevices];
  d_template_channel2 = new agcmplx*[ndevices];
  d_template_channel3 = new agcmplx*[ndevices];

  d_data_freqs = new double*[ndevices];

  d_data_channel1 = new agcmplx*[ndevices];
  d_data_channel2 = new agcmplx*[ndevices];
  d_data_channel3 = new agcmplx*[ndevices];

  d_channel1_ASDinv = new double*[ndevices];
  d_channel2_ASDinv = new double*[ndevices];
  d_channel3_ASDinv = new double*[ndevices];

  d_pHM_trans = new PhenomHMStorage*[ndevices];

  d_pAmp_trans = new IMRPhenomDAmplitudeCoefficients*[ndevices];

  d_amp_prefactors_trans = new AmpInsPrefactors*[ndevices];

  d_pDPreComp_all_trans = new PhenDAmpAndPhasePreComp*[ndevices];

  d_q_all_trans = new HMPhasePreComp*[ndevices];

  d_t0 = new double*[ndevices];

  d_phi0 = new double*[ndevices];

  d_amp0 = new double*[ndevices];

  d_M_tot_sec = new double*[ndevices];

  d_cShift = new double*[ndevices];

  d_inc = new double*[ndevices];
  d_lam = new double*[ndevices];
  d_beta = new double*[ndevices];
  d_psi = new double*[ndevices];
  d_t0_epoch = new double*[ndevices];
  d_tRef_wave_frame = new double*[ndevices];
  d_tRef_sampling_frame = new double*[ndevices];
  d_merger_freq = new double*[ndevices];
  d_phiRef = new double*[ndevices];

  handle = new cublasHandle_t[ndevices];

  for (int i=0; i<ndevices; i++){
      cudaSetDevice(i);
      d_mode_vals[i] = gpu_create_modes(num_modes, nwalkers, l_vals, m_vals, max_length_init, to_gpu, 1);
      gpuErrchk(cudaMalloc(&d_freqs[i], nwalkers*max_length_init*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_H[i], 9*num_modes*nwalkers*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_B[i], 8*max_length_init*num_modes*nwalkers*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_template_channel1[i], data_stream_length*nwalkers*sizeof(agcmplx)));
      gpuErrchk(cudaMalloc(&d_template_channel2[i], data_stream_length*nwalkers*sizeof(agcmplx)));
      gpuErrchk(cudaMalloc(&d_template_channel3[i], data_stream_length*nwalkers*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_freqs[i], data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_data_channel1[i], data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel2[i], data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel3[i], data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_channel1_ASDinv[i], data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel2_ASDinv[i], data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel3_ASDinv[i], data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_pHM_trans[i], nwalkers*sizeof(PhenomHMStorage)));

      gpuErrchk(cudaMalloc(&d_pAmp_trans[i], nwalkers*sizeof(IMRPhenomDAmplitudeCoefficients)));

      gpuErrchk(cudaMalloc(&d_amp_prefactors_trans[i], nwalkers*sizeof(AmpInsPrefactors)));

      gpuErrchk(cudaMalloc(&d_pDPreComp_all_trans[i], num_modes*nwalkers*sizeof(PhenDAmpAndPhasePreComp)));

      gpuErrchk(cudaMalloc(&d_q_all_trans[i], num_modes*nwalkers*sizeof(HMPhasePreComp)));

      gpuErrchk(cudaMalloc(&d_t0[i], nwalkers*sizeof(double)));

      //double phi0_;
      gpuErrchk(cudaMalloc(&d_phi0[i], nwalkers*sizeof(double)));

      //double amp0_;
      gpuErrchk(cudaMalloc(&d_amp0[i], nwalkers*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_M_tot_sec[i], nwalkers*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_cShift[i], 7*sizeof(double)));
      gpuErrchk(cudaMemcpy(d_cShift[i], cShift, 7*sizeof(double), cudaMemcpyHostToDevice));

      gpuErrchk(cudaMalloc(&d_inc[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_lam[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_beta[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_psi[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_t0_epoch[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_tRef_wave_frame[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_tRef_sampling_frame[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_merger_freq[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_phiRef[i], nwalkers*sizeof(double)));

      // for likelihood
      // --------------
      stat = cublasCreate(&handle[i]);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }
          // ----------------

    // initialize values needed for GPU waveform creation
      //double t0_;

      // alocate GPU arrays for interpolation
      interp[i].alloc_arrays(max_length_init, 8*num_modes*nwalkers, d_B[i]);
  }

  #else
  B = new double[8*max_length_init*num_modes*nwalkers];
  interp[0].alloc_arrays(max_length_init, 8*num_modes*nwalkers, B);

  template_channel1 = new agcmplx[data_stream_length*nwalkers];
  template_channel2 = new agcmplx[data_stream_length*nwalkers];
  template_channel3 = new agcmplx[data_stream_length*nwalkers];

  h_data_channel1 = new agcmplx[data_stream_length];
  h_data_channel2 = new agcmplx[data_stream_length];
  h_data_channel3 = new agcmplx[data_stream_length];

  #endif


  PhenomHM::input_data(data_freqs, data_channel1,
                        data_channel2, data_channel3,
                        channel1_ASDinv, channel2_ASDinv,
                        channel3_ASDinv, data_stream_length);
}


void PhenomHM::input_data(double *data_freqs_, cmplx *data_channel1_,
                          cmplx *data_channel2_, cmplx *data_channel3_,
                          double *channel1_ASDinv_, double *channel2_ASDinv_,
                          double *channel3_ASDinv_, int data_stream_length_){

    assert(data_stream_length_ == data_stream_length);

    #if __CUDACC__
    for (int i=0; i<ndevices; i++){
        cudaSetDevice(i);
        gpuErrchk(cudaMemcpy(d_data_freqs[i], data_freqs_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_data_channel1[i], data_channel1_, data_stream_length*sizeof(agcmplx), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_data_channel2[i], data_channel2_, data_stream_length*sizeof(agcmplx), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_data_channel3[i], data_channel3_, data_stream_length*sizeof(agcmplx), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_channel1_ASDinv[i], channel1_ASDinv_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_channel2_ASDinv[i], channel2_ASDinv_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_channel3_ASDinv[i], channel3_ASDinv_, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));
    }

    #else
    memcpy(h_data_channel1, data_channel1_, data_stream_length*sizeof(agcmplx));
    memcpy(h_data_channel2, data_channel2_, data_stream_length*sizeof(agcmplx));
    memcpy(h_data_channel3, data_channel3_, data_stream_length*sizeof(agcmplx));

    channel1_ASDinv = channel1_ASDinv_;
    channel2_ASDinv = channel2_ASDinv_;
    channel3_ASDinv = channel3_ASDinv_;
    #endif
}

/*
generate gpu amp and phase
*/
void PhenomHM::gen_amp_phase(double *freqs_, int current_length_,
    double* m1_, //solar masses
    double* m2_, //solar masses
    double* chi1z_,
    double* chi2z_,
    double* distance_,
    double* phiRef_,
    double* f_ref_){

    assert(current_length_ <= nwalkers*max_length_init);

    freqs = freqs_;
    //printf("fsss: %e, %e\n", freqs[1], freqs[0]);
    d_log10f = log10(freqs[1]) - log10(freqs[0]);
    current_length = current_length_;
    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    phiRef = phiRef_;
    f_ref = f_ref_;

    int i, th_id, nthreads;
    #pragma omp parallel for
    for (int i=0; i<ndevices*nwalkers; i+=1){
            PhenomHM::gen_amp_phase_prep(i, &freqs[i*current_length], current_length_,
                m1_[i], //solar masses
                m2_[i], //solar masses
                chi1z_[i],
                chi2z_[i],
                distance_[i],
                phiRef_[i],
                f_ref_[i]);

            M_tot_sec[i] = (m1[i]+m2[i])*MTSUN_SI;
      }

    #ifdef __CUDACC__
    /* main: evaluate model at given frequencies on GPU */
    NUM_THREADS = 256;

    num_blocks = std::ceil((current_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_blocks, num_modes, nwalkers);
    //printf("%d walkers \n", nwalkers);

    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);

            // copy everything to GPU
            gpuErrchk(cudaMemcpy(d_freqs[i], &freqs[i*nwalkers*current_length], nwalkers*current_length*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_pHM_trans[i], &pHM_trans[i*nwalkers], nwalkers*sizeof(PhenomHMStorage), cudaMemcpyHostToDevice));

            //printf("%.12e, %.12e, %.12e, %.12e\n\n", pHM_trans[2].m1, pHM_trans[2].m2, m1[2], m2[2]);

            gpuErrchk(cudaMemcpy(d_pAmp_trans[i], &pAmp_trans[i*nwalkers], nwalkers*sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_amp_prefactors_trans[i], &amp_prefactors_trans[i*nwalkers], nwalkers*sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans[i], &pDPreComp_all_trans[i*nwalkers*num_modes], nwalkers*num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_q_all_trans[i], &q_all_trans[i*nwalkers*num_modes], nwalkers*num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_t0[i], &t0[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_phi0[i], &phi0[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_amp0[i], &amp0[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_M_tot_sec[i], &M_tot_sec[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            kernel_calculate_all_modes_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals[i],
                  d_pHM_trans[i],
                  d_freqs[i],
                  d_M_tot_sec[i],
                  d_pAmp_trans[i],
                  d_amp_prefactors_trans[i],
                  d_pDPreComp_all_trans[i],
                  d_q_all_trans[i],
                  d_amp0[i],
                  num_modes,
                  d_t0[i],
                  d_phi0[i],
                  d_cShift[i],
                  nwalkers,
                  current_length
              );
              cudaDeviceSynchronize();
              gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_calculate_all_modes_wrap(mode_vals,
          pHM_trans,
          freqs,
          M_tot_sec,
          pAmp_trans,
          amp_prefactors_trans,
          pDPreComp_all_trans,
          q_all_trans,
          amp0,
          num_modes,
          t0,
          phi0,
          cShift,
          nwalkers,
          current_length
      );
    // TODO: add open mp for cpu stuff

    //printf("intrinsic: %e, %e, %e, %e, %e, %e, %e\n", m1, m2, chi1z, chi2z, distance, phiRef, f_ref);

    #endif

     // ensure calls are run in correct order
     current_status = 1;
}


/*
generate structures for GPU creation of amp and phase
*/
void PhenomHM::gen_amp_phase_prep(int ind_walker, double *freqs_gen, int current_length,
    double m1_gen, //solar masses
    double m2_gen, //solar masses
    double chi1z_gen,
    double chi2z_gen,
    double distance_gen,
    double phiRef_gen,
    double f_ref_gen){

    double m1_SI, m2_SI, deltaF;

    // for phenomHM internal calls
    deltaF = -1.0;
    for (int i=0; i<num_modes; i++){
        mode_vals[ind_walker*num_modes + i].length = current_length;
    }

    m1_SI = m1_gen*MSUN_SI;
    m2_SI = m2_gen*MSUN_SI;

    /* fill all the structures necessary for waveform creation */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs_gen,
        current_length,
        m1_SI,
        m2_SI,
        chi1z_gen,
        chi2z_gen,
        distance_gen,
        phiRef_gen,
        deltaF,
        f_ref_gen,
        num_modes,
        to_gpu,
        &pHM_trans[ind_walker],
        &pAmp_trans[ind_walker],
        &amp_prefactors_trans[ind_walker],
        &pDPreComp_all_trans[ind_walker*num_modes],
        &q_all_trans[ind_walker*num_modes],
        &t0[ind_walker],
        &phi0[ind_walker],
        &amp0[ind_walker]);
    //assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in
      //printf("%d, %.12e, %.12e, %.12e, %.12e, %.12e, %.12e, %.12e\n\n", ind_walker, pHM_trans[ind_walker].m1, pHM_trans[ind_walker].m2, m1[ind_walker], m2[ind_walker], t0[ind_walker], phi0[ind_walker], amp0[ind_walker]);

}


/*
Setup interpolation of amp and phase
*/
void PhenomHM::setup_interp_wave(){

    assert(current_status >= 2);

    #ifdef __CUDACC__
    dim3 waveInterpDim(num_blocks, num_modes*nwalkers);

    int i, th_id, nthreads;
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            // fill B array
            fill_B_wave_wrap<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_B[i], current_length, num_modes*nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            // perform interpolation
            interp[i].prep(d_B[i], current_length, 2*num_modes*nwalkers, 1);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            set_spline_constants_wave_wrap<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_B[i], current_length, num_modes*nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_fill_B_wave_wrap(mode_vals, B, current_length, num_modes*nwalkers);

    // perform interpolation
    interp[0].prep(B, current_length, 2*num_modes*nwalkers, 0);

    cpu_set_spline_constants_wave_wrap(mode_vals, B, current_length, num_modes*nwalkers);
    #endif

    if (current_status == 2) current_status = 3;
}

/*
Get LISA fast Fourier domain response on GPU
*/

void PhenomHM::LISAresponseFD(double* inc_, double* lam_, double* beta_, double* psi_, double* t0_epoch_, double* tRef_wave_frame_, double* tRef_sampling_frame_, double* merger_freq_){
    inc = inc_;
    lam = lam_;
    beta = beta_;
    psi = psi_;
    t0_epoch = t0_epoch_;
    tRef_wave_frame = tRef_wave_frame_;
    tRef_sampling_frame = tRef_sampling_frame_;
    merger_freq = merger_freq_;

    //printf("extrinsic: %e, %e, %e, %e, %e, %e, %e\n", inc, lam, beta, psi, t0_epoch, tRef_wave_frame, tRef_sampling_frame, merger_freq);

    assert(current_status >= 1);
    // get H on the CPU
    int i, th_id, nthreads;
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices*nwalkers; i+=nthreads){
            prep_H_info(&H[i*num_modes*9], l_vals, m_vals, num_modes, inc[i], lam[i], beta[i], psi[i], phiRef[i]);
        }
    }

    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    #ifdef __CUDACC__
    int num_blocks = std::ceil((current_length + NUM_THREADS - 1)/NUM_THREADS);
    dim3 gridDim(num_blocks, num_modes, nwalkers);

    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            gpuErrchk(cudaMemcpy(d_H[i], &H[i*9*num_modes*nwalkers], 9*num_modes*nwalkers*sizeof(agcmplx), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_inc[i], &inc[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_lam[i], &lam[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_beta[i], &beta[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_psi[i], &psi[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_phiRef[i], &phiRef[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_t0_epoch[i], &t0_epoch[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_tRef_wave_frame[i], &tRef_wave_frame[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_tRef_sampling_frame[i], &tRef_sampling_frame[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_merger_freq[i], &merger_freq[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            int num_blocks = std::ceil((current_length + NUM_THREADS - 1)/NUM_THREADS);

            // Perform response

            kernel_JustLISAFDresponseTDI_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals[i], d_H[i], d_freqs[i], d_freqs[i], d_log10f, d_l_vals, d_m_vals,
                        num_modes, current_length, d_inc[i], d_lam[i], d_beta[i], d_psi[i], d_phiRef[i], d_t0_epoch[i],
                        d_tRef_wave_frame[i], d_tRef_sampling_frame[i], d_merger_freq[i], TDItag, 0, nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            kernel_add_tRef_phase_shift_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals[i], d_freqs[i],
                        num_modes, current_length, d_tRef_wave_frame[i], nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_JustLISAFDresponseTDI_wrap(mode_vals, (agcmplx*)H, freqs, freqs, d_log10f, l_vals, m_vals,
                num_modes, current_length, inc, lam, beta, psi, phiRef, t0_epoch,
                tRef_wave_frame, tRef_sampling_frame, merger_freq, TDItag, 0, nwalkers);

    cpu_add_tRef_phase_shift_wrap(mode_vals, freqs,
                num_modes, current_length, tRef_wave_frame, nwalkers);

    #endif

    if (current_status == 1) current_status = 2;
}

/*
setup interpolation for the response transfer functions
*/
void PhenomHM::setup_interp_response(){

    assert(current_status >= 3);

    #ifdef __CUDACC__
    dim3 responseInterpDim(num_blocks, num_modes*nwalkers);

    int i, th_id, nthreads;
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            fill_B_response_wrap<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_B[i], current_length, num_modes*nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            interp[i].prep(d_B[i], current_length, 8*num_modes*nwalkers, 1);  // TODO check the 8?
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            set_spline_constants_response_wrap<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_B[i], current_length, num_modes*nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_fill_B_response_wrap(mode_vals, B, current_length, num_modes*nwalkers);

    interp[0].prep(B, current_length, 8*num_modes*nwalkers, 0);  // TODO check the 8?

    cpu_set_spline_constants_response_wrap(mode_vals, B, current_length, num_modes*nwalkers);
    #endif
    if (current_status == 3) current_status = 4;
}

/*
interpolate amp and phase up to frequencies of the data stream.
*/
void PhenomHM::perform_interp(){
    assert(current_status >= 4);

    #ifdef __CUDACC__
    int num_block_interp = std::ceil((data_stream_length + NUM_THREADS - 1)/NUM_THREADS);
    //dim3 mainInterpDim(num_block_interp, 1, nwalkers);//, num_modes);
    dim3 mainInterpDim(num_block_interp, 1, nwalkers);//, num_modes);

    int i, th_id, nthreads;
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            interpolate_wrap<<<mainInterpDim, NUM_THREADS>>>(d_template_channel1[i], d_template_channel2[i], d_template_channel3[i], d_mode_vals[i], num_modes,
                d_log10f, d_freqs[i], current_length, d_data_freqs[i], data_stream_length, d_t0_epoch[i],
                d_tRef_sampling_frame[i], d_channel1_ASDinv[i], d_channel2_ASDinv[i], d_channel3_ASDinv[i], t_obs_start, t_obs_end, nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }

    #else
    cpu_interpolate_wrap(template_channel1, template_channel2, template_channel3, mode_vals, num_modes,
        d_log10f, freqs, current_length, data_freqs, data_stream_length, t0_epoch,
        tRef_sampling_frame, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, t_obs_start, t_obs_end, nwalkers);
    #endif

    if (current_status == 4) current_status = 5;
}

/*
Compute likelihood on the GPU
*/
void PhenomHM::Likelihood (double *d_h_arr, double *h_h_arr){

    //printf("like mem\n");
    //print_mem_info();
    #ifdef __CUDACC__
    gpu_likelihood(d_h_arr, h_h_arr, d_data_channel1, d_data_channel2, d_data_channel3,
                                    d_template_channel1, d_template_channel2, d_template_channel3,
                                   data_stream_length, nwalkers, ndevices, handle);

    #else
    cpu_likelihood(d_h_arr, h_h_arr, h_data_channel1, h_data_channel2, h_data_channel3,
                                    template_channel1, template_channel2, template_channel3,
                                   data_stream_length, nwalkers, ndevices);

    #endif
     assert(current_status == 5);

    //}
}

/*
Copy TDI channels to CPU and return to python.
*/
void PhenomHM::GetTDI (cmplx* channel1_, cmplx* channel2_, cmplx* channel3_) {

  assert(current_status > 4);

  #ifdef __CUDACC__
  for (int i=0; i<ndevices; i++){
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(&channel1_[i*nwalkers*data_stream_length], d_template_channel1[i], data_stream_length*nwalkers*sizeof(cmplx), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&channel2_[i*nwalkers*data_stream_length], d_template_channel2[i], data_stream_length*nwalkers*sizeof(cmplx), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&channel3_[i*nwalkers*data_stream_length], d_template_channel3[i], data_stream_length*nwalkers*sizeof(cmplx), cudaMemcpyDeviceToHost));
  }
  #else

  memcpy(channel1_, template_channel1, data_stream_length*nwalkers*sizeof(cmplx));
  memcpy(channel2_, template_channel2, data_stream_length*nwalkers*sizeof(cmplx));
  memcpy(channel3_, template_channel3, data_stream_length*nwalkers*sizeof(cmplx));
  #endif
}

/*
auxillary function for getting amplitude and phase to the CPU
*/
#ifdef __CUDACC__
CUDA_KERNEL void read_out_amp_phase(ModeContainer *mode_vals, double *amp, double *phase, int num_modes, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.y;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    amp[mode_i*length + i] = mode_vals[mode_i].amp[i];
    phase[mode_i*length + i] = mode_vals[mode_i].phase[i];
}
#endif

/*
Return amplitude and phase in python on CPU
*/
void PhenomHM::GetAmpPhase(double* amp_, double* phase_) {
  assert(current_status >= 1);

  #ifdef __CUDACC__
  double *amp, *phase;

  dim3 readOutDim(num_blocks, num_modes*nwalkers);
  for (int i=0; i<ndevices; i++){
      cudaSetDevice(i);
      gpuErrchk(cudaMalloc(&amp, nwalkers*num_modes*current_length*sizeof(double)));
      gpuErrchk(cudaMalloc(&phase, nwalkers*num_modes*current_length*sizeof(double)));
      read_out_amp_phase<<<readOutDim, NUM_THREADS>>>(d_mode_vals[i], amp, phase, nwalkers*num_modes, current_length);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());
      gpuErrchk(cudaMemcpy(&amp_[i*nwalkers*num_modes*current_length], amp, nwalkers*num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&phase_[i*nwalkers*num_modes*current_length], phase, nwalkers*num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk( cudaFree(amp));
      gpuErrchk(cudaFree(phase));
  }
  #else
  for (int walker_i=0; walker_i<nwalkers; walker_i++){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
      memcpy(&amp_[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].amp, current_length*sizeof(double));
      memcpy(&phase_[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].phase, current_length*sizeof(double));
  }
}
  #endif
}

/*
Destructor
*/
PhenomHM::~PhenomHM() {
  delete[] pHM_trans;
  delete[] pAmp_trans;
  delete[] amp_prefactors_trans;
  delete[] pDPreComp_all_trans;
  delete[] q_all_trans;
  delete[] t0;
  delete[] amp0;
  delete[] phi0;
  delete[] M_tot_sec;
  cpu_destroy_modes(mode_vals);
  delete[] H;
  delete[] cShift;

  #ifdef __CUDACC__
  gpuErrchk(cudaFree(d_data_freqs));
  for (int i=0; i<ndevices; i++){
      gpuErrchk(cudaFree(d_freqs[i]));
      gpu_destroy_modes(d_mode_vals[i]);

      gpuErrchk(cudaFree(d_pHM_trans[i]));
      gpuErrchk(cudaFree(d_pAmp_trans[i]));
      gpuErrchk(cudaFree(d_amp_prefactors_trans[i]));
      gpuErrchk(cudaFree(d_pDPreComp_all_trans[i]));
      gpuErrchk(cudaFree(d_q_all_trans[i]));
      gpuErrchk(cudaFree(d_cShift[i]));

      gpuErrchk(cudaFree(d_data_channel1[i]));
      gpuErrchk(cudaFree(d_data_channel2[i]));
      gpuErrchk(cudaFree(d_data_channel3[i]));

      gpuErrchk(cudaFree(d_template_channel1[i]));
      gpuErrchk(cudaFree(d_template_channel2[i]));
      gpuErrchk(cudaFree(d_template_channel3));

      gpuErrchk(cudaFree(d_channel1_ASDinv[i]));
      gpuErrchk(cudaFree(d_channel2_ASDinv[i]));
      gpuErrchk(cudaFree(d_channel3_ASDinv[i]));
      cublasDestroy(handle[i]);
      gpuErrchk(cudaFree(d_B[i]));
      gpuErrchk(cudaFree(d_t0[i]));
      gpuErrchk(cudaFree(d_phi0[i]));
      gpuErrchk(cudaFree(d_amp0[i]));
      gpuErrchk(cudaFree(d_M_tot_sec[i]));

      gpuErrchk(cudaFree(d_inc[i]));
      gpuErrchk(cudaFree(d_lam[i]));
      gpuErrchk(cudaFree(d_beta[i]));
      gpuErrchk(cudaFree(d_psi[i]));
      gpuErrchk(cudaFree(d_t0_epoch[i]));
      gpuErrchk(cudaFree(d_tRef_wave_frame[i]));
      gpuErrchk(cudaFree(d_tRef_sampling_frame[i]));
      gpuErrchk(cudaFree(d_merger_freq[i]));
      gpuErrchk(cudaFree(d_phiRef[i]));
  }

  delete[] d_freqs;
  delete[] d_mode_vals;

  delete[] d_pHM_trans;
  delete[] d_pAmp_trans;
  delete[] d_amp_prefactors_trans;
  delete[] d_pDPreComp_all_trans;
  delete[] d_q_all_trans;
  delete[] d_cShift;

  delete[] d_data_channel1;
  delete[] d_data_channel2;
  delete[] d_data_channel3;

  delete[] d_template_channel1;
  delete[] d_template_channel2;
  delete[] d_template_channel3;

  delete[] d_channel1_ASDinv;
  delete[] d_channel2_ASDinv;
  delete[] d_channel3_ASDinv;
  delete[] handle;
  delete[] d_B;
  delete[] d_t0;
  delete[] d_phi0;
  delete[] d_amp0;
  delete[] d_M_tot_sec;

  delete[] d_inc;
  delete[] d_lam;
  delete[] d_beta;
  delete[] d_psi;
  delete[] d_t0_epoch;
  delete[] d_tRef_wave_frame;
  delete[] d_tRef_sampling_frame;
  delete[] d_merger_freq;
  delete[] d_phiRef;

  delete[] handle;

  #else
  delete[] B;

  delete[] template_channel1;
  delete[] template_channel2;
  delete[] template_channel3;

  delete[] h_data_channel1;
  delete[] h_data_channel2;
  delete[] h_data_channel3;

  #endif

  delete[] interp;
}

int GetDeviceCount(){
    int num_device_check;
    #ifdef __CUDACC__
    cudaError_t cuda_status = cudaGetDeviceCount(&num_device_check);
    if (cudaSuccess != cuda_status) num_device_check = 0;
    #else
    num_device_check = 0;
    #endif
    printf("NUMBER OF DEVICES: %d\n", num_device_check);
    return num_device_check;
}
