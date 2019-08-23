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

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"
#include "cublas_v2.h"
#include "interpolate.cu"
#include "fdresponse.h"
#include "createGPUHolders.cu"
#include "kernel_response.cu"
// TODO: CUTOFF PHASE WHEN IT STARTS TO GO BACK UP!!!

using namespace std;

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
    double t_obs_dur_){

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

    TDItag = TDItag_;
    t_obs_dur = t_obs_dur_;
    to_gpu = 1;

    cudaError_t err;

    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

    // malloc and setup for the GPU

  gpuErrchk(cudaMalloc(&d_B, 8*data_stream_length_*num_modes*sizeof(double)));

  mode_vals = cpu_create_modes(num_modes, l_vals, m_vals, max_length_init, to_gpu, 1);

  gpuErrchk(cudaMalloc(&d_H, 9*num_modes*sizeof(cuDoubleComplex)));

  gpuErrchk(cudaMalloc(&d_template_channel1, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMalloc(&d_template_channel2, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMalloc(&d_template_channel3, data_stream_length*sizeof(cuDoubleComplex)));

  d_mode_vals = gpu_create_modes(num_modes, l_vals, m_vals, max_length_init, to_gpu, 1);

  gpuErrchk(cudaMalloc(&d_freqs, max_length_init*sizeof(double)));

  gpuErrchk(cudaMalloc(&d_data_freqs, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_data_freqs, data_freqs, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_data_channel1, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMemcpy(d_data_channel1, data_channel1, data_stream_length*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_data_channel2, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMemcpy(d_data_channel2, data_channel2, data_stream_length*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_data_channel3, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMemcpy(d_data_channel3, data_channel3, data_stream_length*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));


  gpuErrchk(cudaMalloc(&d_channel1_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_channel1_ASDinv, channel1_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_channel2_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_channel2_ASDinv, channel2_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_channel3_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_channel3_ASDinv, channel3_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_pHM_trans, sizeof(PhenomHMStorage)));

  gpuErrchk(cudaMalloc(&d_pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients)));

  gpuErrchk(cudaMalloc(&d_amp_prefactors_trans, sizeof(AmpInsPrefactors)));

  gpuErrchk(cudaMalloc(&d_pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp)));

  gpuErrchk(cudaMalloc((void**) &d_q_all_trans, num_modes*sizeof(HMPhasePreComp)));

  // phase shifts for each m mode
  double cShift[7] = {0.0,
                       PI_2 /* i shift */,
                       0.0,
                       -PI_2 /* -i shift */,
                       PI /* 1 shift */,
                       PI_2 /* -1 shift */,
                       0.0};

  gpuErrchk(cudaMalloc(&d_cShift, 7*sizeof(double)));

  gpuErrchk(cudaMemcpy(d_cShift, &cShift, 7*sizeof(double), cudaMemcpyHostToDevice));


  // for likelihood
  // --------------
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("CUBLAS initialization failed\n");
          exit(0);
      }
      // ----------------

// initialize values needed for GPU waveform creation
  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;

  // alocate for H matrix from pyFDResponse in LDC
  H = new cmplx[9*num_modes];

  // alocate GPU arrays for interpolation
  interp.alloc_arrays(max_length_init);
}

/*
generate gpu amp and phase
*/
void PhenomHM::gen_amp_phase(double *freqs_, int current_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double phiRef_,
    double f_ref_){

    assert(to_gpu == 1);
    assert(current_length_ <= max_length_init);

    PhenomHM::gen_amp_phase_prep(freqs_, current_length_,
        m1_, //solar masses
        m2_, //solar masses
        chi1z_,
        chi2z_,
        distance_,
        phiRef_,
        f_ref_);

    freqs = freqs_;
    current_length = current_length_;
    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    phiRef = phiRef_;
    f_ref = f_ref_;
    //printf("intrinsic: %e, %e, %e, %e, %e, %e, %e\n", m1, m2, chi1z, chi2z, distance, phiRef, f_ref);

    // copy everything to GPU
    gpuErrchk(cudaMemcpy(d_freqs, freqs, current_length*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pHM_trans, pHM_trans, sizeof(PhenomHMStorage), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pAmp_trans, pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_amp_prefactors_trans, amp_prefactors_trans, sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans, pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_q_all_trans, q_all_trans, num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice));

    double M_tot_sec = (m1+m2)*MTSUN_SI;
    /* main: evaluate model at given frequencies on GPU */
    NUM_THREADS = 256;
    num_blocks = std::ceil((current_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_blocks, num_modes);
    kernel_calculate_all_modes<<<gridDim, NUM_THREADS>>>(d_mode_vals,
          d_pHM_trans,
          d_freqs,
          M_tot_sec,
          d_pAmp_trans,
          d_amp_prefactors_trans,
          d_pDPreComp_all_trans,
          d_q_all_trans,
          amp0,
          num_modes,
          t0,
          phi0,
          d_cShift
      );
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

     // ensure calls are run in correct order
     current_status = 1;
}

/*
generate structures for GPU creation of amp and phase
*/
void PhenomHM::gen_amp_phase_prep(double *freqs, int current_length,
    double m1, //solar masses
    double m2, //solar masses
    double chi1z,
    double chi2z,
    double distance,
    double phiRef,
    double f_ref){

    // for phenomHM internal calls
    deltaF = -1.0;

    for (int i=0; i<num_modes; i++){
        mode_vals[i].length = current_length;
    }

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* fill all the structures necessary for waveform creation */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        current_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        phiRef,
        deltaF,
        f_ref,
        num_modes,
        to_gpu,
        pHM_trans,
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in
}

/*
Setup interpolation of amp and phase
*/
void PhenomHM::setup_interp_wave(){

    assert(current_status >= 2);
    dim3 waveInterpDim(num_blocks, num_modes);

    // fill B array
    fill_B_wave<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // perform interpolation
    interp.prep(d_B, current_length, 2*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_wave<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 2) current_status = 3;
}

/*
Get LISA fast Fourier domain response on GPU
*/
void PhenomHM::LISAresponseFD(double inc_, double lam_, double beta_, double psi_, double t0_epoch_, double tRef_wave_frame_, double tRef_sampling_frame_, double merger_freq_){
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
    prep_H_info(H, l_vals, m_vals, num_modes, inc, lam, beta, psi, phiRef);
    gpuErrchk(cudaMemcpy(d_H, H, 9*num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    int num_blocks = std::ceil((current_length + NUM_THREADS - 1)/NUM_THREADS);
    dim3 gridDim(num_blocks, num_modes);

    //printf("inc %lf, beta %lf, lam %lf, psi %lf, phiRef %e, t0_epoch %lf, tRef %lf\n", inc, beta, lam, psi, phiRef, t0_epoch, tRef);
    // Perform response
    kernel_JustLISAFDresponseTDI_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals, d_H, d_freqs, d_freqs, d_log10f, d_l_vals, d_m_vals, num_modes, current_length, inc, lam, beta, psi, phiRef, t0_epoch, tRef_wave_frame, tRef_sampling_frame, merger_freq, TDItag, 0);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 1) current_status = 2;
}

/*
setup interpolation for the response transfer functions
*/
void PhenomHM::setup_interp_response(){

    assert(current_status >= 3);

    dim3 responseInterpDim(num_blocks, num_modes);

    fill_B_response<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, current_length, 8*num_modes, 1);  // TODO check the 8?
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_response<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 3) current_status = 4;
}

/*
interpolate amp and phase up to frequencies of the data stream.
*/
void PhenomHM::perform_interp(){
    assert(current_status >= 4);
    int num_block_interp = std::ceil((data_stream_length + NUM_THREADS - 1)/NUM_THREADS);
    dim3 mainInterpDim(num_block_interp);//, num_modes);
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    interpolate<<<mainInterpDim, NUM_THREADS>>>(d_template_channel1, d_template_channel2, d_template_channel3, d_mode_vals, num_modes, d_log10f, d_freqs, current_length, d_data_freqs, data_stream_length, t0_epoch, tRef_sampling_frame, d_channel1_ASDinv, d_channel2_ASDinv, d_channel3_ASDinv, t_obs_dur);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 4) current_status = 5;
}

/*
Compute likelihood on the GPU
*/
void PhenomHM::Likelihood (double *like_out_){

     assert(current_status == 5);
     double d_h = 0.0;
     double h_h = 0.0;
     char * status;
     double res;
     cuDoubleComplex result;

        // get data - template terms
         stat = cublasZdotc(handle, data_stream_length,
                 d_template_channel1, 1,
                 d_data_channel1, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);
         //printf("channel1 d_h: %e\n", cuCreal(result));

         stat = cublasZdotc(handle, data_stream_length,
                 d_template_channel2, 1,
                 d_data_channel2, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);
         //printf("channel2 d_h: %e\n", cuCreal(result));

         stat = cublasZdotc(handle, data_stream_length,
                 d_template_channel3, 1,
                 d_data_channel3, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);
         //printf("channel3 d_h: %e\n", cuCreal(result));


         // get template template terms
        stat = cublasZdotc(handle, data_stream_length,
                     d_template_channel1, 1,
                     d_template_channel1, 1,
                     &result);
             status = _cudaGetErrorEnum(stat);
              cudaDeviceSynchronize();

              if (stat != CUBLAS_STATUS_SUCCESS) {
                      exit(0);
                  }
             h_h += cuCreal(result);
             //printf("channel1 h_h: %e\n", cuCreal(result));

             stat = cublasZdotc(handle, data_stream_length,
                     d_template_channel2, 1,
                     d_template_channel2, 1,
                     &result);
             status = _cudaGetErrorEnum(stat);
              cudaDeviceSynchronize();

              if (stat != CUBLAS_STATUS_SUCCESS) {
                      exit(0);
                  }
             h_h += cuCreal(result);
             //printf("channel2 h_h: %e\n", cuCreal(result));

             stat = cublasZdotc(handle, data_stream_length,
                     d_template_channel3, 1,
                     d_template_channel3, 1,
                     &result);
             status = _cudaGetErrorEnum(stat);
              cudaDeviceSynchronize();

              if (stat != CUBLAS_STATUS_SUCCESS) {
                      exit(0);
                  }
             h_h += cuCreal(result);
             //printf("channel3 h_h: %e\n", cuCreal(result));

     like_out_[0] = 4*d_h;
     like_out_[1] = 4*h_h;
}

/*
Copy TDI channels to CPU and return to python.
*/
void PhenomHM::GetTDI (cmplx* channel1_, cmplx* channel2_, cmplx* channel3_) {

  assert(current_status > 4);
  gpuErrchk(cudaMemcpy(channel1_, d_template_channel1, data_stream_length*sizeof(cmplx), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(channel2_, d_template_channel2, data_stream_length*sizeof(cmplx), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(channel3_, d_template_channel3, data_stream_length*sizeof(cmplx), cudaMemcpyDeviceToHost));
}

/*
auxillary function for getting amplitude and phase to the CPU
*/
__global__ void read_out_amp_phase(ModeContainer *mode_vals, double *amp, double *phase, int num_modes, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.y;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    amp[mode_i*length + i] = mode_vals[mode_i].amp[i];
    phase[mode_i*length + i] = mode_vals[mode_i].phase[i];
}

/*
Return amplitude and phase in python on CPU
*/
void PhenomHM::GetAmpPhase(double* amp_, double* phase_) {
  assert(current_status > 1);
  double *amp, *phase;
  gpuErrchk(cudaMalloc(&amp, num_modes*current_length*sizeof(double)));
  gpuErrchk(cudaMalloc(&phase, num_modes*current_length*sizeof(double)));

  dim3 readOutDim(num_blocks, num_modes);
  read_out_amp_phase<<<readOutDim, NUM_THREADS>>>(d_mode_vals, amp, phase, num_modes, current_length);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaMemcpy(amp_, amp, num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(phase_, phase, num_modes*current_length*sizeof(double), cudaMemcpyDeviceToHost));

  gpuErrchk( cudaFree(amp));
  gpuErrchk(cudaFree(phase));
}

/*
Destructor
*/
PhenomHM::~PhenomHM() {
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete[] pDPreComp_all_trans;
  delete[] q_all_trans;
  cpu_destroy_modes(mode_vals);
  delete[] H;

  gpuErrchk(cudaFree(d_freqs));
  gpuErrchk(cudaFree(d_data_freqs));
  gpu_destroy_modes(d_mode_vals);
  gpuErrchk(cudaFree(d_pHM_trans));
  gpuErrchk(cudaFree(d_pAmp_trans));
  gpuErrchk(cudaFree(d_amp_prefactors_trans));
  gpuErrchk(cudaFree(d_pDPreComp_all_trans));
  gpuErrchk(cudaFree(d_q_all_trans));
  gpuErrchk(cudaFree(d_cShift));

  gpuErrchk(cudaFree(d_data_channel1));
  gpuErrchk(cudaFree(d_data_channel2));
  gpuErrchk(cudaFree(d_data_channel3));

  gpuErrchk(cudaFree(d_template_channel1));
  gpuErrchk(cudaFree(d_template_channel2));
  gpuErrchk(cudaFree(d_template_channel3));

  gpuErrchk(cudaFree(d_channel1_ASDinv));
  gpuErrchk(cudaFree(d_channel2_ASDinv));
  gpuErrchk(cudaFree(d_channel3_ASDinv));
  cublasDestroy(handle);
  gpuErrchk(cudaFree(d_B));
}
