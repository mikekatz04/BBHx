/*
This is the central piece of code. This file implements a class
that takes data in on the cpu side, copies
it to the gpu, and exposes functions that let
you perform actions with the GPU

This class will get translated into python via cython
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


using namespace std;

PhenomHM::PhenomHM (int max_length_init_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    double *data_freqs_,
    cmplx *data_stream_, int data_stream_length_, double *X_ASDinv_, double *Y_ASDinv_, double *Z_ASDinv_){

    max_length_init = max_length_init_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    data_freqs = data_freqs_;
    data_stream = data_stream_;
    data_stream_length = data_stream_length_;
    X_ASDinv = X_ASDinv_;
    Y_ASDinv = Y_ASDinv_;
    Z_ASDinv = Z_ASDinv_;

    to_gpu = 1;

    cudaError_t err;

    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

  gpuErrchk(cudaMalloc(&d_B, 7*data_stream_length_*num_modes*sizeof(double)));

  mode_vals = cpu_create_modes(num_modes, l_vals, m_vals, max_length_init, to_gpu, 1);

  gpuErrchk(cudaMalloc(&d_H, 9*num_modes*sizeof(cuDoubleComplex)));

  gpuErrchk(cudaMalloc(&d_X, data_stream_length*num_modes*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMalloc(&d_Y, data_stream_length*num_modes*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMalloc(&d_Z, data_stream_length*num_modes*sizeof(cuDoubleComplex)));

  d_mode_vals = gpu_create_modes(num_modes, l_vals, m_vals, max_length_init, to_gpu, 1);

  gpuErrchk(cudaMalloc(&d_freqs, max_length_init*sizeof(double)));

  gpuErrchk(cudaMalloc(&d_data_freqs, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_data_freqs, data_freqs, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_data_stream, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMemcpy(d_data_stream, data_stream, data_stream_length*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_X_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_X_ASDinv, X_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_Y_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_Y_ASDinv, Y_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_Z_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_Z_ASDinv, Z_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_pHM_trans, sizeof(PhenomHMStorage)));

  gpuErrchk(cudaMalloc(&d_pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients)));

  gpuErrchk(cudaMalloc(&d_amp_prefactors_trans, sizeof(AmpInsPrefactors)));

  gpuErrchk(cudaMalloc(&d_pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp)));

  gpuErrchk(cudaMalloc((void**) &d_q_all_trans, num_modes*sizeof(HMPhasePreComp)));


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

  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;

  H = new cmplx[9*num_modes];
}


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

    gpuErrchk(cudaMemcpy(d_freqs, freqs, current_length*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pHM_trans, pHM_trans, sizeof(PhenomHMStorage), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pAmp_trans, pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_amp_prefactors_trans, amp_prefactors_trans, sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans, pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_q_all_trans, q_all_trans, num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice));

    double M_tot_sec = (m1+m2)*MTSUN_SI;
    /* main: evaluate model at given frequencies */
    NUM_THREADS = 256;
    num_blocks = std::ceil((current_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_blocks, num_modes);
    //printf("blocks %d\n", num_blocks);
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

    /* main: evaluate model at given frequencies */
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


void PhenomHM::setup_interp_wave(){

    assert(current_status >= 1);
    dim3 waveInterpDim(num_blocks, num_modes);

    fill_B_wave<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, current_length, 2*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_wave<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 1) current_status = 2;
}

void PhenomHM::LISAresponseFD(double inc_, double lam_, double beta_, double psi_, double t0_epoch_, double tRef_, double merger_freq_, int TDItag_){
    inc = inc_;
    lam = lam_;
    beta = beta_;
    psi = psi_;
    t0_epoch = t0_epoch_;
    tRef = tRef_;
    TDItag = TDItag_;
    merger_freq = merger_freq_;

    assert(current_status >= 2);

    prep_H_info(H, l_vals, m_vals, num_modes, inc, lam, beta, psi, phiRef);
    gpuErrchk(cudaMemcpy(d_H, H, 9*num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    int num_blocks = std::ceil((current_length + NUM_THREADS - 1)/NUM_THREADS);
    dim3 gridDim(num_blocks, num_modes);

    kernel_JustLISAFDresponseTDI_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals, d_H, d_freqs, d_freqs, d_log10f, d_l_vals, d_m_vals, num_modes, current_length, inc, lam, beta, psi, phiRef, t0_epoch, tRef, merger_freq, TDItag, 0);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 2) current_status = 3;
}

void PhenomHM::setup_interp_response(){

    assert(current_status >= 3);

    dim3 responseInterpDim(num_blocks, num_modes);

    fill_B_response<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, current_length, 7*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_response<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, current_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 3) current_status = 4;
}

void PhenomHM::perform_interp(double f_min, double df, int length_new){
    assert(current_status >= 4);
    assert(length_new == data_stream_length);
    int num_block_interp = std::ceil((length_new + NUM_THREADS - 1)/NUM_THREADS);
    dim3 mainInterpDim(num_block_interp, num_modes);
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    interpolate<<<mainInterpDim, NUM_THREADS>>>(d_X, d_Y, d_Z, d_mode_vals, num_modes, f_min, df, d_log10f, d_freqs, current_length, length_new, t0, tRef, d_X_ASDinv, d_Y_ASDinv, d_Z_ASDinv);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    if (current_status == 4) current_status = 5;
}

void PhenomHM::Likelihood (double *like_out_){

     assert(current_status == 5);
     double d_h = 0.0;
     double h_h = 0.0;
     char * status;
     double res;
     cuDoubleComplex result;
     for (int mode_i=0; mode_i<num_modes; mode_i++){
         stat = cublasZdotc(handle, data_stream_length,
                 &d_X[mode_i*data_stream_length], 1,
                 d_data_stream, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);

         stat = cublasZdotc(handle, data_stream_length,
                 &d_Y[mode_i*data_stream_length], 1,
                 d_data_stream, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);

         stat = cublasZdotc(handle, data_stream_length,
                 &d_Z[mode_i*data_stream_length], 1,
                 d_data_stream, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);
     }

     // d_X d_X for h_h
     stat = cublasDznrm2(handle, num_modes*data_stream_length,
             d_X, 1, &res);
     status = _cudaGetErrorEnum(stat);
      cudaDeviceSynchronize();

      if (stat != CUBLAS_STATUS_SUCCESS) {
              exit(0);
          }
        h_h += res*res; //TODO: MAKE SURE THIS IS RIGHT

      // d_Y d_Y for h_h
      stat = cublasDznrm2(handle, num_modes*data_stream_length,
              d_Y, 1, &res);
      status = _cudaGetErrorEnum(stat);
       cudaDeviceSynchronize();

       if (stat != CUBLAS_STATUS_SUCCESS) {
               exit(0);
           }
         h_h += res*res;

       // d_Z d_Z for h_h
       stat = cublasDznrm2(handle, num_modes*data_stream_length,
               d_Z, 1, &res);
       status = _cudaGetErrorEnum(stat);
        cudaDeviceSynchronize();

        if (stat != CUBLAS_STATUS_SUCCESS) {
                exit(0);
            }
     h_h += res*res;

     like_out_[0] = d_h;
     like_out_[1] = h_h;
}

void PhenomHM::GetTDI (cmplx* X_, cmplx* Y_, cmplx* Z_) {

  assert(current_status > 4);
  gpuErrchk(cudaMemcpy(X_, d_X, data_stream_length*num_modes*sizeof(cmplx), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Y_, d_Y, data_stream_length*num_modes*sizeof(cmplx), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Z_, d_Z, data_stream_length*num_modes*sizeof(cmplx), cudaMemcpyDeviceToHost));
}

__global__ void read_out_amp_phase(ModeContainer *mode_vals, double *amp, double *phase, int num_modes, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.y;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    amp[mode_i*length + i] = mode_vals[mode_i].amp[i];
    phase[mode_i*length + i] = mode_vals[mode_i].phase[i];
}

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
  gpuErrchk(cudaFree(d_data_stream));
  gpu_destroy_modes(d_mode_vals);
  gpuErrchk(cudaFree(d_pHM_trans));
  gpuErrchk(cudaFree(d_pAmp_trans));
  gpuErrchk(cudaFree(d_amp_prefactors_trans));
  gpuErrchk(cudaFree(d_pDPreComp_all_trans));
  gpuErrchk(cudaFree(d_q_all_trans));
  gpuErrchk(cudaFree(d_cShift));
  gpuErrchk(cudaFree(d_X));
  gpuErrchk(cudaFree(d_Y));
  gpuErrchk(cudaFree(d_Z));
  gpuErrchk(cudaFree(d_X_ASDinv));
  gpuErrchk(cudaFree(d_Y_ASDinv));
  gpuErrchk(cudaFree(d_Z_ASDinv));
  cublasDestroy(handle);
  gpuErrchk(cudaFree(d_B));
}
