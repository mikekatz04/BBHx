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

#include "createGPUHolders.hh"
#include <cuda_runtime_api.h>
#include <cuda.h>

#endif

#include "IMRPhenomD.h"
#include <d_kernel.hh>
#include "kernel_response.hh"
#include "interpolate.hh"
#include <likelihood.hh>

#include <d_manager.hh>
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

#ifdef __CUDACC__
__global__
void print_it()
{
  print_stuff();
}

#endif

PhenomD::PhenomD (int max_length_init_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    int data_stream_length_,
    int TDItag_,
    double t_obs_start_,
    double t_obs_end_,
    int nwalkers_,
    int ndevices_){

    #pragma omp parallel
    {
      if (omp_get_thread_num() == 1) printf("NUM OMP THREADS: %d\n", omp_get_num_threads());
    }

    #ifdef __CUDACC__
    printf("try to print\n");
    print_it<<<1,1>>>();
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    printf("past printing\n");
    #endif

    max_length_init = max_length_init_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    data_stream_length = data_stream_length_;

    nwalkers = nwalkers_;

    TDItag = TDItag_;
    t_obs_start = t_obs_start_;
    t_obs_end = t_obs_end_;

    data_added = 0;

    #ifdef __GLOBAL_FIT__
    int is_global_fit = 1;
    #else
    int is_global_fit = 0;
    #endif

    #ifdef __CUDACC__
    to_gpu = 1;

    #else
    to_gpu = 0;
    assert(ndevices == 1);
    #endif

    ndevices = ndevices_;


    // DECLARE ALL THE  NECESSARY STRUCTS

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes*nwalkers*ndevices];

    t0 = new double[nwalkers*ndevices];

    phi0 = new double[nwalkers*ndevices];

    amp0 = new double[nwalkers*ndevices];

    // malloc and setup for the GPU

  mode_vals = cpu_create_modes(num_modes, nwalkers*ndevices, l_vals, m_vals, max_length_init, to_gpu, 1);

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
  d_upper_diag = new double*[ndevices];
  d_diag = new double*[ndevices];
  d_lower_diag = new double*[ndevices];

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

  d_pDPreComp_all_trans = new PhenDAmpAndPhasePreComp*[ndevices];

  d_t0 = new double*[ndevices];
  d_f_ref = new double*[ndevices];

  d_phi0 = new double*[ndevices];

  d_amp0 = new double*[ndevices];

  d_M_tot_sec = new double*[ndevices];

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
  int wtf = 0;
  for (int i=0; i<ndevices; i++){
      cudaSetDevice(i);
      //d_mode_vals[i] = gpu_create_modes(num_modes, nwalkers, l_vals, m_vals, max_length_init, to_gpu, 1);
      gpuErrchk(cudaMalloc(&d_freqs[i], nwalkers*max_length_init*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_H[i], 9*num_modes*nwalkers*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_B[i], 8*max_length_init*num_modes*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_upper_diag[i], 8*max_length_init*num_modes*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_diag[i], 8*max_length_init*num_modes*nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_lower_diag[i], 8*max_length_init*num_modes*nwalkers*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_data_freqs[i], data_stream_length*sizeof(double)));

      #ifdef __GLOBAL_FIT__
      #else

        gpuErrchk(cudaMalloc(&d_template_channel1[i], data_stream_length*nwalkers*sizeof(agcmplx)));
        gpuErrchk(cudaMalloc(&d_template_channel2[i], data_stream_length*nwalkers*sizeof(agcmplx)));
        gpuErrchk(cudaMalloc(&d_template_channel3[i], data_stream_length*nwalkers*sizeof(agcmplx)));


      gpuErrchk(cudaMalloc(&d_data_channel1[i], data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel2[i], data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_data_channel3[i], data_stream_length*sizeof(agcmplx)));

      gpuErrchk(cudaMalloc(&d_channel1_ASDinv[i], data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel2_ASDinv[i], data_stream_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_channel3_ASDinv[i], data_stream_length*sizeof(double)));

      #endif

      gpuErrchk(cudaMalloc(&d_pDPreComp_all_trans[i], num_modes*nwalkers*sizeof(PhenDAmpAndPhasePreComp)));

      gpuErrchk(cudaMalloc(&d_t0[i], nwalkers*sizeof(double)));
      gpuErrchk(cudaMalloc(&d_f_ref[i], nwalkers*sizeof(double)));

      //double phi0_;
      gpuErrchk(cudaMalloc(&d_phi0[i], nwalkers*sizeof(double)));

      //double amp0_;
      gpuErrchk(cudaMalloc(&d_amp0[i], nwalkers*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_M_tot_sec[i], nwalkers*sizeof(double)));

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
  upper_diag = new double[8*max_length_init*num_modes*nwalkers];
  diag = new double[8*max_length_init*num_modes*nwalkers];
  lower_diag = new double[8*max_length_init*num_modes*nwalkers];
  interp[0].alloc_arrays(max_length_init, 8*num_modes*nwalkers, B);

  #ifdef __GLOBAL_FIT__
  #else
  template_channel1 = new agcmplx[data_stream_length*nwalkers];
  template_channel2 = new agcmplx[data_stream_length*nwalkers];
  template_channel3 = new agcmplx[data_stream_length*nwalkers];

  h_data_channel1 = new agcmplx[data_stream_length];
  h_data_channel2 = new agcmplx[data_stream_length];
  h_data_channel3 = new agcmplx[data_stream_length];
  #endif // __GLOBAL_FIT__
  #endif //__CUDACC__
}

void PhenomD::input_global_data(long ptr_data_freqs_,
                                  long ptr_template_channel1_,
                            long ptr_template_channel2_, long ptr_template_channel3_, int data_stream_length_)

{
  #ifdef __GLOBAL_FIT__
  #else
  printf("Cannot input global data if not working with global fit. Need to use input_data.\n");
  assert(0);
  #endif

  assert(data_stream_length_ == data_stream_length);

  #ifdef __CUDACC__
  d_data_freqs[0] = (double *) ptr_data_freqs_;
  d_template_channel1[0] = (agcmplx*) ptr_template_channel1_;
  d_template_channel2[0] = (agcmplx*) ptr_template_channel2_;
  d_template_channel3[0] = (agcmplx*) ptr_template_channel3_;

  if (ndevices > 1){
      printf("not implemnted yet\n");
      assert(0);
  }

  #else
  data_freqs = (double *) ptr_data_freqs_;
  template_channel1 = (agcmplx*) ptr_template_channel1_;
  template_channel2 = (agcmplx*) ptr_template_channel2_;
  template_channel3 = (agcmplx*) ptr_template_channel3_;
  #endif

  data_added = 1;
}


void PhenomD::input_data(double *data_freqs_, cmplx *data_channel1_,
                          cmplx *data_channel2_, cmplx *data_channel3_,
                          double *channel1_ASDinv_, double *channel2_ASDinv_,
                          double *channel3_ASDinv_, int data_stream_length_){

    #ifdef __GLOBAL_FIT__
    printf("Cannot input data if working with global fit. Need to use input_global_data.\n");
    assert(0);
    #endif

    assert(data_stream_length_ == data_stream_length);

    data_freqs = data_freqs_;
    channel1_ASDinv = channel1_ASDinv_;
    channel2_ASDinv = channel2_ASDinv_;
    channel3_ASDinv = channel3_ASDinv_;
    data_channel1 = data_channel1_;
    data_channel2 = data_channel2_;
    data_channel3 = data_channel3_;

    #ifdef __CUDACC__
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
    #endif // __CUDACC__

    data_added = 1;
}


/*
generate gpu amp and phase
*/
void PhenomD::gen_amp_phase(double *freqs_, int current_length_,
    double* m1_, //solar masses
    double* m2_, //solar masses
    double* chi1z_,
    double* chi2z_,
    double* distance_,
    double* phiRef_,
    double* f_ref_, double* inc_, double* lam_, double* beta_, double* psi_,
    double* t0_epoch_, double* tRef_wave_frame_, double* tRef_sampling_frame_, double* merger_freq_){
        inc = inc_;
        lam = lam_;
        beta = beta_;
        psi = psi_;
        t0_epoch = t0_epoch_;
        tRef_wave_frame = tRef_wave_frame_;
        tRef_sampling_frame = tRef_sampling_frame_;
        merger_freq = merger_freq_;

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
            PhenomD::gen_amp_phase_prep(i, &freqs[i*current_length], current_length_,
                m1_[i], //solar masses
                m2_[i], //solar masses
                chi1z_[i],
                chi2z_[i],
                distance_[i],
                phiRef_[i],
                f_ref_[i]);
      }

      #pragma omp parallel private(th_id, i)
      {
      //for (int i=0; i<nwalkers; i++){
          nthreads = omp_get_num_threads();
          th_id = omp_get_thread_num();
          for (int i=th_id; i<ndevices*nwalkers; i+=nthreads){
              prep_H_info(&H[i*num_modes*9], l_vals, m_vals, num_modes, inc[i], lam[i], beta[i], psi[i], phiRef[i]);
          }
      }

    #ifdef __CUDACC__
    /* main: evaluate model at given frequencies on GPU */
    NUM_THREADS = 256;

    num_blocks = std::ceil((current_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_blocks, 1, nwalkers);

    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);

            // copy everything to GPU
            gpuErrchk(cudaMemcpy(d_f_ref[i], &f_ref[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));
            //gpuErrchk(cudaMemcpy(d_freqs[i], &freqs[i*nwalkers*current_length], nwalkers*current_length*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans[i], &pDPreComp_all_trans[i*nwalkers], nwalkers*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_t0[i], &t0[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_phi0[i], &phi0[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_amp0[i], &amp0[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_M_tot_sec[i], &M_tot_sec[i*nwalkers], nwalkers*sizeof(double), cudaMemcpyHostToDevice));

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


            kernel_calculate_all_modes_PhenomD<<<gridDim, NUM_THREADS>>>(d_mode_vals[i],
                  d_data_freqs[i],
                  d_M_tot_sec[i],
                  d_pDPreComp_all_trans[i],
                  d_amp0[i],
                  d_t0[i],
                  d_phi0[i],
                  current_length,
          				d_f_ref[i],
                  nwalkers,
                  d_H[i], d_data_freqs[i], d_data_freqs[i], d_log10f, d_l_vals, d_m_vals,
                              num_modes, current_length, d_inc[i], d_lam[i], d_beta[i], d_psi[i], d_phiRef[i], d_t0_epoch[i],
                              d_tRef_wave_frame[i], d_tRef_sampling_frame[i], d_merger_freq[i], TDItag, 0, nwalkers, d_M_tot_sec[i], d_pDPreComp_all_trans[i], t_obs_end*YRSID_SI, d_t0[i],
                              d_template_channel1[i], d_template_channel2[i], d_template_channel3[i], d_channel1_ASDinv[i],
                                  d_channel2_ASDinv[i], d_channel3_ASDinv[i]
                );
              cudaDeviceSynchronize();
              gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_calculate_all_modes_PhenomD(mode_vals,
          data_freqs,
          M_tot_sec,
          pDPreComp_all_trans,
          amp0,
          t0,
          phi0,
          current_length,
          f_ref,
          nwalkers,
          (agcmplx*)H, data_freqs, data_freqs, d_log10f, l_vals, m_vals,
                      num_modes, current_length, inc, lam, beta, psi, phiRef, t0_epoch,
                      tRef_wave_frame, tRef_sampling_frame, merger_freq, TDItag, 0, nwalkers, M_tot_sec, pDPreComp_all_trans, t_obs_end*YRSID_SI, t0,
                      template_channel1, template_channel2, template_channel3, channel1_ASDinv,
                          channel2_ASDinv, channel3_ASDinv
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
void PhenomD::gen_amp_phase_prep(int ind_walker, double *freqs_gen, int current_length,
    double m1_gen, //solar masses
    double m2_gen, //solar masses
    double chi1z_gen,
    double chi2z_gen,
    double distance_gen,
    double phiRef_gen,
    double f_ref_gen){

    double m1_SI, m2_SI, deltaF;

    // for PhenomD internal calls
    deltaF = -1.0;
    mode_vals[ind_walker].length = current_length;

    m1_SI = m1_gen*MSUN_SI;
    m2_SI = m2_gen*MSUN_SI;

    /* fill all the structures necessary for waveform creation */
    retcode = 0;
    retcode = ins_IMRPhenomDSetupAmpAndPhaseCoefficients(&pDPreComp_all_trans[ind_walker],
    m1_SI,
    m2_SI,
    chi1z_gen,
    chi2z_gen);

    UsefulPowers powers_of_f;

    PNPhasingSeries pn = pDPreComp_all_trans[ind_walker].pn;
    IMRPhenomDPhaseCoefficients pPhi = pDPreComp_all_trans[ind_walker].pPhi;
    PhiInsPrefactors phi_prefactors = pDPreComp_all_trans[ind_walker].phi_prefactors;
    IMRPhenomDAmplitudeCoefficients pAmp = pDPreComp_all_trans[ind_walker].pAmp;
    AmpInsPrefactors amp_prefactors = pDPreComp_all_trans[ind_walker].amp_prefactors;

    M_tot_sec[ind_walker] = (m1_gen+m1_gen)*MTSUN_SI;
    double Mf = f_ref_gen*M_tot_sec[ind_walker];
    int status_in_for = init_useful_powers(&powers_of_f, Mf);

    amp0[ind_walker] = PhenomUtilsFDamp0(m1_gen + m2_gen, distance_gen);
    double phi_22_at_f_ref = PhiInsAnsatzInt(Mf, &powers_of_f, &phi_prefactors, &pPhi, &pn);
    phi0[ind_walker] = 0.5 * (phi_22_at_f_ref);
    t0[ind_walker] = -1./(2.0*PI)*DPhiInsAnsatzInt(Mf, &pPhi, &pn)*M_tot_sec[ind_walker];
}


CUDA_CALLABLE_MEMBER
void combine_it_inner(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers, int walker_i, int i){

  double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
  double time_check, amp, phase, phaseRdelay;
  double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
  agcmplx ampphasefactor;
  agcmplx I = agcmplx(0.0, 1.0);
  int old_ind_below;
  agcmplx trans_complex;
  time_check  = old_mode_vals[walker_i].time_freq_corr[i];

  if (time_check == -1.0) {
      return;
  }

  //printf("%d, %e, %e, %e\n", i, data_freqs[i], old_mode_vals[walker_i].time_freq_corr[i], old_mode_vals[walker_i].amp[i]);
  amp = old_mode_vals[walker_i].amp[i];

  if (amp < 1e-40){
      return;
  }

  phase  = old_mode_vals[walker_i].phase[i];

  phaseRdelay  = old_mode_vals[walker_i].phaseRdelay[i];
  ampphasefactor = amp*gcmplx::exp(agcmplx(0.0, phase + phaseRdelay));

  // X or A

  transferL1_re  = old_mode_vals[walker_i].transferL1_re[i];

  transferL1_im = old_mode_vals[walker_i].transferL1_im[i];

  trans_complex = agcmplx(transferL1_re, transferL1_im)* ampphasefactor * channel1_ASDinv[i]; //TODO may be faster to load as complex number with 0.0 for imaginary part


  channel1_out[walker_i*data_length + i] += trans_complex;

  // Y or E

  transferL2_re  = old_mode_vals[walker_i].transferL2_re[i];
  transferL2_im  = old_mode_vals[walker_i].transferL2_im[i];

  trans_complex = agcmplx(transferL2_re, transferL2_im)* ampphasefactor * channel2_ASDinv[i]; //TODO may be faster to load as complex number with 0.0 for imaginary part

  channel2_out[walker_i*data_length + i] += trans_complex;

  // Z or T
  transferL3_re  = old_mode_vals[walker_i].transferL3_re[i];

  transferL3_im  = old_mode_vals[walker_i].transferL3_im[i];

  trans_complex = agcmplx(transferL3_re, transferL3_im)* ampphasefactor * channel3_ASDinv[i]; //TODO may be faster to load as complex number with 0.0 for imaginary part

  channel3_out[walker_i*data_length + i] += trans_complex;

}

#ifdef __CUDACC__
CUDA_KERNEL
void gpu_combine_it(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers)
{


    //#pragma omp parallel for collapse(2)
    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < num_walkers;
         walker_i += blockDim.z * gridDim.z){

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < data_length;
         i += blockDim.x * gridDim.x){
          combine_it_inner(channel1_out, channel2_out, channel3_out, old_mode_vals,
              num_modes, d_log10f, old_freqs, old_length, data_freqs, data_length, t0_arr, tRef_arr, channel1_ASDinv,
              channel2_ASDinv, channel3_ASDinv, t_obs_start, t_obs_end, num_walkers, walker_i, i);

  }
}
}

#else

void cpu_combine_it(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers)
{
  double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
  double time_check, amp, phase, phaseRdelay;
  double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
  agcmplx ampphasefactor;
  agcmplx I = agcmplx(0.0, 1.0);
  int old_ind_below;
  agcmplx trans_complex;

    //#pragma omp parallel for collapse(2)
    for (int walker_i=0; walker_i<num_walkers; walker_i++){
      for (int i=0; i<data_length; i++){
          combine_it_inner(channel1_out, channel2_out, channel3_out, old_mode_vals,
              num_modes, d_log10f, old_freqs, old_length, data_freqs, data_length, t0_arr, tRef_arr, channel1_ASDinv,
              channel2_ASDinv, channel3_ASDinv, t_obs_start, t_obs_end, num_walkers, walker_i, i);

  }
}
}

#endif

void PhenomD::Combine(){
    #ifdef __CUDACC__
    NUM_THREADS = 256;
    int i, th_id, nthreads;
    num_blocks = std::ceil((current_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_blocks, 1, nwalkers);
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            gpu_combine_it<<<gridDim, NUM_THREADS>>>(d_template_channel1[i], d_template_channel2[i], d_template_channel3[i], d_mode_vals[i], num_modes,
                d_log10f, d_freqs[i], current_length, d_data_freqs[i], data_stream_length, d_t0_epoch[i],
                d_tRef_sampling_frame[i], d_channel1_ASDinv[i], d_channel2_ASDinv[i], d_channel3_ASDinv[i], t_obs_start, t_obs_end, nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
      }
    }
    #else
    cpu_combine_it(template_channel1, template_channel2, template_channel3, mode_vals, num_modes,
        d_log10f, freqs, current_length, data_freqs, data_stream_length, t0_epoch,
        tRef_sampling_frame, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, t_obs_start, t_obs_end, nwalkers);

    #endif
}

/*
Setup interpolation of amp and phase
*/
void PhenomD::setup_interp_wave(){

    //assert(current_status >= 2);

    #ifdef __CUDACC__
    dim3 waveInterpDim(num_blocks, num_modes, nwalkers);

    int i, th_id, nthreads;
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            // fill B array
            fill_B_wave_wrap<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_freqs[i],
                                                             d_B[i], d_upper_diag[i],
                                                             d_diag[i], d_lower_diag[i],
                                                             current_length, num_modes, nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            // perform interpolation
            interp[i].prep(d_B[i], d_upper_diag[i], d_diag[i], d_lower_diag[i], current_length, 2*num_modes*nwalkers, 1);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            set_spline_constants_wave_wrap<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_B[i], current_length, nwalkers, num_modes, d_freqs[i]);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_fill_B_wave_wrap(mode_vals, freqs, B, upper_diag, diag, lower_diag, current_length, num_modes, nwalkers);

    // perform interpolation
    interp[0].prep(B, upper_diag, diag, lower_diag, current_length, 2*num_modes*nwalkers, 0);

    cpu_set_spline_constants_wave_wrap(mode_vals, B, current_length, nwalkers, num_modes, freqs);
    #endif

    if (current_status == 2) current_status = 3;
}

/*
Get LISA fast Fourier domain response on GPU
*/

void PhenomD::LISAresponseFD(double* inc_, double* lam_, double* beta_, double* psi_, double* t0_epoch_, double* tRef_wave_frame_, double* tRef_sampling_frame_, double* merger_freq_){
    inc = inc_;
    lam = lam_;
    beta = beta_;
    psi = psi_;
    t0_epoch = t0_epoch_;
    tRef_wave_frame = tRef_wave_frame_;
    tRef_sampling_frame = tRef_sampling_frame_;
    merger_freq = merger_freq_;

    //printf("extrinsic: %e, %e, %e, %e, %e, %e, %e\n", inc, lam, beta, psi, t0_epoch, tRef_wave_frame, tRef_sampling_frame, merger_freq);

    //assert(current_status >= 1);
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

            //kernel_add_tRef_phase_shift_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals[i], d_freqs[i],
            //            num_modes, current_length, d_tRef_wave_frame[i], nwalkers);
            //cudaDeviceSynchronize();
            //gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_JustLISAFDresponseTDI_wrap(mode_vals, (agcmplx*)H, freqs, freqs, d_log10f, l_vals, m_vals,
                num_modes, current_length, inc, lam, beta, psi, phiRef, t0_epoch,
                tRef_wave_frame, tRef_sampling_frame, merger_freq, TDItag, 0, nwalkers);

    //cpu_add_tRef_phase_shift_wrap(mode_vals, freqs,
    //            num_modes, current_length, tRef_wave_frame, nwalkers);

    #endif

    if (current_status == 1) current_status = 2;
}

/*
setup interpolation for the response transfer functions
*/
void PhenomD::setup_interp_response(){

    //assert(current_status >= 3);

    #ifdef __CUDACC__
    dim3 responseInterpDim(num_blocks, num_modes, nwalkers);

    int i, th_id, nthreads;
    #pragma omp parallel private(th_id, i)
    {
    //for (int i=0; i<nwalkers; i++){
        nthreads = omp_get_num_threads();
        th_id = omp_get_thread_num();
        for (int i=th_id; i<ndevices; i+=nthreads){
            cudaSetDevice(i);
            fill_B_response_wrap<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_freqs[i],
                                                                     d_B[i], d_upper_diag[i],
                                                                     d_diag[i], d_lower_diag[i],
                                                                     current_length, num_modes, nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            interp[i].prep(d_B[i], d_upper_diag[i], d_diag[i], d_lower_diag[i], current_length, 8*num_modes*nwalkers, 1);  // TODO check the 8?
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());

            set_spline_constants_response_wrap<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals[i], d_B[i], current_length, nwalkers, num_modes, d_freqs[i]);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }
    #else
    cpu_fill_B_response_wrap(mode_vals, freqs, B, upper_diag, diag, lower_diag, current_length, num_modes, nwalkers);

    interp[0].prep(B, upper_diag, diag, lower_diag, current_length, 8*num_modes*nwalkers, 0);  // TODO check the 8?

    cpu_set_spline_constants_response_wrap(mode_vals, B, current_length, nwalkers, num_modes, freqs);
    #endif
    if (current_status == 3) current_status = 4;
}

/*
interpolate amp and phase up to frequencies of the data stream.
*/
void PhenomD::perform_interp(){
    //assert(current_status >= 4);
    assert(data_added == 1);
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
                d_tRef_sampling_frame[i], d_tRef_wave_frame[i], d_channel1_ASDinv[i], d_channel2_ASDinv[i], d_channel3_ASDinv[i], t_obs_start, t_obs_end, nwalkers);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
        }
    }

    #else
    cpu_interpolate_wrap(template_channel1, template_channel2, template_channel3, mode_vals, num_modes,
        d_log10f, freqs, current_length, data_freqs, data_stream_length, t0_epoch,
        tRef_sampling_frame, tRef_wave_frame, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, t_obs_start, t_obs_end, nwalkers);
    #endif

    if (current_status == 4) current_status = 5;
}

#ifdef __CUDACC__
__global__
void gpu_reset_to_zero(int length, agcmplx *template_channel1, agcmplx *template_channel2, agcmplx *template_channel3){

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < length;
       i += blockDim.x * gridDim.x){
         template_channel1[i] = agcmplx(0.0, 0.0);
           template_channel2[i] = agcmplx(0.0, 0.0);
             template_channel3[i] = agcmplx(0.0, 0.0);
  }
}
#else
void cpu_reset_to_zero(int length, agcmplx *template_channel1, agcmplx *template_channel2, agcmplx *template_channel3){
  for (int i = 0;
       i < length;
       i += 1){
         template_channel1[i] = agcmplx(0.0, 0.0);
           template_channel2[i] = agcmplx(0.0, 0.0);
             template_channel3[i] = agcmplx(0.0, 0.0);
  }
}
#endif

void PhenomD::ResetGlobalTemplate(){

  #ifdef __CUDACC__
  int NUM_THREADS = 256;
  int num_block_interp = std::ceil((data_stream_length + NUM_THREADS - 1)/NUM_THREADS);
  dim3 resetDim(num_block_interp, 1, 1);
  gpu_reset_to_zero<<<resetDim, NUM_THREADS>>>(data_stream_length, d_template_channel1[0], d_template_channel2[0], d_template_channel3[0]);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  #else

 cpu_reset_to_zero(data_stream_length, template_channel1, template_channel2, template_channel3);
 #endif

}

/*
Compute likelihood on the GPU
*/
void PhenomD::Likelihood (double *d_h_arr, double *h_h_arr){
    #ifdef __GLOBAL_FIT__
    printf("With global fit, need to control likelihood from exterior class.\n");
    assert(0);
    #endif

    assert(data_added == 1);
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
     //assert(current_status == 5);

    //}
}

/*
Copy TDI channels to CPU and return to python.
*/
void PhenomD::GetTDI (cmplx* channel1_, cmplx* channel2_, cmplx* channel3_) {

  //assert(current_status > 4);

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
CUDA_KERNEL void read_out_response(ModeContainer *mode_vals, agcmplx *transferL1, agcmplx *transferL2, agcmplx *transferL3,
                                    double* phaseRdelay, double *time_freq_corr, int num_modes, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.y;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    transferL1[mode_i*length + i] = agcmplx(mode_vals[mode_i].transferL1_re[i], mode_vals[mode_i].transferL1_im[i]);
    transferL2[mode_i*length + i] = agcmplx(mode_vals[mode_i].transferL2_re[i], mode_vals[mode_i].transferL2_im[i]);
    transferL3[mode_i*length + i] = agcmplx(mode_vals[mode_i].transferL3_re[i], mode_vals[mode_i].transferL3_im[i]);
    phaseRdelay[mode_i*length + i] = mode_vals[mode_i].phaseRdelay[i];
    time_freq_corr[mode_i*length + i] = mode_vals[mode_i].time_freq_corr[i];

}
#endif

/*
Return amplitude and phase in python on CPU
*/
void PhenomD::GetResponse(cmplx* transferL1_, cmplx* transferL2_, cmplx* transferL3_, double* phaseRdelay_, double* time_freq_corr_) {
  //assert(current_status >= 1);

  #ifdef __CUDACC__
  agcmplx *transferL1, *transferL2, *transferL3;
  double *phaseRdelay, *time_freq_corr;

  dim3 readOutDim(num_blocks, num_modes*nwalkers);
  for (int i=0; i<ndevices; i++){
      cudaSetDevice(i);
      gpuErrchk(cudaMalloc(&transferL1, nwalkers*num_modes*current_length*sizeof(agcmplx)));
      gpuErrchk(cudaMalloc(&transferL2, nwalkers*num_modes*current_length*sizeof(agcmplx)));
      gpuErrchk(cudaMalloc(&transferL3, nwalkers*num_modes*current_length*sizeof(agcmplx)));
      gpuErrchk(cudaMalloc(&phaseRdelay, nwalkers*num_modes*current_length*sizeof(double)));
      gpuErrchk(cudaMalloc(&time_freq_corr, nwalkers*num_modes*current_length*sizeof(double)));
      read_out_response<<<readOutDim, NUM_THREADS>>>(d_mode_vals[i], transferL1, transferL2, transferL3, phaseRdelay, time_freq_corr,
                                                      nwalkers*num_modes, current_length);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());
      gpuErrchk(cudaMemcpy(&transferL1_[i*nwalkers*num_modes*current_length], transferL1, nwalkers*num_modes*current_length*sizeof(agcmplx), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&transferL2_[i*nwalkers*num_modes*current_length], transferL2, nwalkers*num_modes*current_length*sizeof(agcmplx), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&transferL3_[i*nwalkers*num_modes*current_length], transferL3, nwalkers*num_modes*current_length*sizeof(agcmplx), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&phaseRdelay_[i*nwalkers*num_modes*current_length], phaseRdelay, nwalkers*num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&time_freq_corr_[i*nwalkers*num_modes*current_length], time_freq_corr, nwalkers*num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk( cudaFree(transferL1));
      gpuErrchk(cudaFree(transferL2));
      gpuErrchk(cudaFree(transferL3));
      gpuErrchk(cudaFree(phaseRdelay));
      gpuErrchk(cudaFree(time_freq_corr));
  }
  #else
  double *transferL1_re = new double[nwalkers*num_modes*current_length];
  double *transferL1_im = new double[nwalkers*num_modes*current_length];
  double *transferL2_re = new double[nwalkers*num_modes*current_length];
  double *transferL2_im = new double[nwalkers*num_modes*current_length];
  double *transferL3_re = new double[nwalkers*num_modes*current_length];
  double *transferL3_im = new double[nwalkers*num_modes*current_length];

  for (int walker_i=0; walker_i<nwalkers; walker_i++){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
      memcpy(&transferL1_re[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].transferL1_re, current_length*sizeof(double));
      memcpy(&transferL1_im[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].transferL1_im, current_length*sizeof(double));
      memcpy(&transferL2_re[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].transferL2_re, current_length*sizeof(double));
      memcpy(&transferL2_im[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].transferL2_im, current_length*sizeof(double));
      memcpy(&transferL3_re[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].transferL3_re, current_length*sizeof(double));
      memcpy(&transferL3_im[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].transferL3_im, current_length*sizeof(double));

      memcpy(&phaseRdelay_[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].phaseRdelay, current_length*sizeof(double));
      memcpy(&time_freq_corr_[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].time_freq_corr, current_length*sizeof(double));

  }
}

  for (int i=0; i<nwalkers*num_modes*current_length; i++){
      transferL1_[i] = cmplx(transferL1_re[i], transferL1_im[i]);
      transferL2_[i] = cmplx(transferL2_re[i], transferL2_im[i]);
      transferL3_[i] = cmplx(transferL3_re[i], transferL3_im[i]);
  }
  delete[] transferL1_re;
  delete[] transferL1_im;
  delete[] transferL2_re;
  delete[] transferL2_im;
  delete[] transferL3_re;
  delete[] transferL3_im;

  #endif
}

/*
auxillary function for getting amplitude and phase to the CPU
*/
#ifdef __CUDACC__
CUDA_KERNEL void read_out_phase_spline(ModeContainer *mode_vals, double *phase, double *coeff1, double *coeff2, double *coeff3, int num_modes, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.y;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    phase[mode_i*length + i] = mode_vals[mode_i].phase[i];

    if (i >= length-1) return;
    coeff1[mode_i*length + i] = mode_vals[mode_i].phase_coeff_1[i];
    coeff2[mode_i*length + i] = mode_vals[mode_i].phase_coeff_2[i];
    coeff3[mode_i*length + i] = mode_vals[mode_i].phase_coeff_3[i];
}
#endif


/*
Return amplitude and phase in python on CPU
*/
void PhenomD::GetPhaseSpline(double* phase_, double* coeff1_, double* coeff2_, double* coeff3_) {
  //assert(current_status >= 1);

  #ifdef __CUDACC__
  double *phase, *coeff1, *coeff2, *coeff3;

  dim3 readOutDim(num_blocks, num_modes*nwalkers);
  for (int i=0; i<ndevices; i++){
      cudaSetDevice(i);
      gpuErrchk(cudaMalloc(&phase, nwalkers*num_modes*current_length*sizeof(double)));
      gpuErrchk(cudaMalloc(&coeff1, nwalkers*num_modes*(current_length-1)*sizeof(double)));
      gpuErrchk(cudaMalloc(&coeff2, nwalkers*num_modes*(current_length-1)*sizeof(double)));
      gpuErrchk(cudaMalloc(&coeff3, nwalkers*num_modes*(current_length-1)*sizeof(double)));
      read_out_phase_spline<<<readOutDim, NUM_THREADS>>>(d_mode_vals[i], phase, coeff1, coeff2, coeff3, nwalkers*num_modes, current_length);
      cudaDeviceSynchronize();
      gpuErrchk(cudaGetLastError());
      gpuErrchk(cudaMemcpy(&phase_[i*nwalkers*num_modes*current_length], phase, nwalkers*num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&coeff1_[i*nwalkers*num_modes*(current_length-1)], coeff1, nwalkers*num_modes*(current_length-1)*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&coeff2_[i*nwalkers*num_modes*(current_length-1)], coeff2, nwalkers*num_modes*(current_length-1)*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&coeff3_[i*nwalkers*num_modes*(current_length-1)], coeff3, nwalkers*num_modes*(current_length-1)*sizeof(double), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaFree(phase));
      gpuErrchk(cudaFree(coeff1));
      gpuErrchk(cudaFree(coeff2));
      gpuErrchk(cudaFree(coeff3));
  }
  #else
  for (int walker_i=0; walker_i<nwalkers; walker_i++){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
      memcpy(&phase_[walker_i*num_modes*current_length + mode_i*current_length], mode_vals[walker_i*num_modes + mode_i].phase, current_length*sizeof(double));
      memcpy(&coeff1_[walker_i*num_modes*(current_length-1) + mode_i*(current_length-1)], mode_vals[walker_i*num_modes + mode_i].phase_coeff_1, (current_length-1)*sizeof(double));
      memcpy(&coeff2_[walker_i*num_modes*(current_length-1) + mode_i*(current_length-1)], mode_vals[walker_i*num_modes + mode_i].phase_coeff_2, (current_length-1)*sizeof(double));
      memcpy(&coeff3_[walker_i*num_modes*(current_length-1) + mode_i*(current_length-1)], mode_vals[walker_i*num_modes + mode_i].phase_coeff_3, (current_length-1)*sizeof(double));
    }
}
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
void PhenomD::GetAmpPhase(double* amp_, double* phase_) {
  //assert(current_status >= 1);

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
PhenomD::~PhenomD() {
  delete[] pDPreComp_all_trans;
  delete[] t0;
  delete[] amp0;
  delete[] phi0;
  delete[] M_tot_sec;
  cpu_destroy_modes(mode_vals);
  delete[] H;

  #ifdef __CUDACC__
  gpuErrchk(cudaFree(d_data_freqs));
  for (int i=0; i<ndevices; i++){
      gpuErrchk(cudaFree(d_freqs[i]));
      //gpu_destroy_modes(d_mode_vals[i]);

      gpuErrchk(cudaFree(d_pDPreComp_all_trans[i]));

      #ifdef __GLOBAL_FIT__
      #else
      gpuErrchk(cudaFree(d_data_channel1[i]));
      gpuErrchk(cudaFree(d_data_channel2[i]));
      gpuErrchk(cudaFree(d_data_channel3[i]));

      gpuErrchk(cudaFree(d_template_channel1[i]));
      gpuErrchk(cudaFree(d_template_channel2[i]));
      gpuErrchk(cudaFree(d_template_channel3));

      gpuErrchk(cudaFree(d_channel1_ASDinv[i]));
      gpuErrchk(cudaFree(d_channel2_ASDinv[i]));
      gpuErrchk(cudaFree(d_channel3_ASDinv[i]));
      #endif
      cublasDestroy(handle[i]);
      gpuErrchk(cudaFree(d_B[i]));
      gpuErrchk(cudaFree(d_upper_diag[i]));
      gpuErrchk(cudaFree(d_diag[i]));
      gpuErrchk(cudaFree(d_lower_diag[i]));
      gpuErrchk(cudaFree(d_t0[i]));
      gpuErrchk(cudaFree(d_f_ref[i]));
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

  delete[] d_pDPreComp_all_trans;

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
  delete[] d_upper_diag;
  delete[] d_diag;
  delete[] d_lower_diag;
  delete[] d_t0;
  delete[] d_f_ref;
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

  #ifdef __GLOBAL_FIT__
  #else
  delete[] template_channel1;
  delete[] template_channel2;
  delete[] template_channel3;

  delete[] h_data_channel1;
  delete[] h_data_channel2;
  delete[] h_data_channel3;
  #endif

  delete[] upper_diag;
  delete[] lower_diag;
  delete[] diag;

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
