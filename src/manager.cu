/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
//#include <reduction.cu>
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
    std::complex<double> *data_stream_, int data_stream_length_, double *X_ASDinv_, double *Y_ASDinv_, double *Z_ASDinv_){

    max_length_init = max_length_init_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
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

  gpuErrchk(cudaMalloc(&d_data_stream, data_stream_length*sizeof(cuDoubleComplex)));
  gpuErrchk(cudaMemcpy(d_data_stream, data_stream, data_stream_length*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_X_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_X_ASDinv, X_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_Y_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_Y_ASDinv, Y_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_Z_ASDinv, data_stream_length*sizeof(double)));
  gpuErrchk(cudaMemcpy(d_Z_ASDinv, Z_ASDinv, data_stream_length*sizeof(double), cudaMemcpyHostToDevice));

  //gpuErrchk(cudaMalloc(&d_mode_vals, num_modes*sizeof(d_mode_vals)));
  //gpuErrchk(cudaMemcpy(d_mode_vals, mode_vals, num_modes*sizeof(d_mode_vals), cudaMemcpyHostToDevice));

  // DECLARE ALL THE  NECESSARY STRUCTS
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
}


void PhenomHM::gen_amp_phase(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    assert(to_gpu == 1);

    PhenomHM::gen_amp_phase_prep(freqs_, f_length_,
        m1_, //solar masses
        m2_, //solar masses
        chi1z_,
        chi2z_,
        distance_,
        inclination_,
        phiRef_,
        deltaF_,
        f_ref_);

    gpuErrchk(cudaMemcpy(d_freqs, freqs, f_length*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pHM_trans, pHM_trans, sizeof(PhenomHMStorage), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pAmp_trans, pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_amp_prefactors_trans, amp_prefactors_trans, sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans, pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_q_all_trans, q_all_trans, num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice));

    double M_tot_sec = (m1+m2)*MTSUN_SI;
    /* main: evaluate model at given frequencies */
    NUM_THREADS = 256;
    num_blocks = std::ceil((f_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_modes, num_blocks);
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

}

void PhenomHM::gen_amp_phase_prep(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    freqs = freqs_;
    f_length = f_length_;
    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    inclination = inclination_;
    phiRef = phiRef_;
    deltaF = deltaF_;
    f_ref = f_ref_;

    for (int i=0; i<num_modes; i++){
        mode_vals[i].length = f_length;
    }

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        f_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        inclination,
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

    dim3 waveInterpDim(num_modes, num_blocks);

    fill_B_wave<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, f_length, 2*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_wave<<<waveInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
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

    H = prep_H_info(l_vals, m_vals, num_modes, inc, lam, beta, psi, phiRef);
    gpuErrchk(cudaMemcpy(d_H, H, 9*num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    int num_blocks = std::ceil((f_length + NUM_THREADS - 1)/NUM_THREADS);
    dim3 gridDim(num_modes, num_blocks);

    kernel_JustLISAFDresponseTDI_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals, d_H, d_freqs, d_freqs, d_log10f, d_l_vals, d_m_vals, num_modes, f_length, inc, lam, beta, psi, phiRef, t0_epoch, tRef, merger_freq, TDItag, 0);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void PhenomHM::setup_interp_response(){

    dim3 responseInterpDim(num_modes, num_blocks);

    fill_B_response<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, f_length, 7*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_response<<<responseInterpDim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void PhenomHM::perform_interp(double f_min, double df, int length_new){
    int num_block_interp = std::ceil((length_new + NUM_THREADS - 1)/NUM_THREADS);
    dim3 mainInterpDim(num_modes, num_block_interp);
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    interpolate<<<mainInterpDim, NUM_THREADS>>>(d_X, d_Y, d_Z, d_mode_vals, num_modes, f_min, df, d_log10f, d_freqs, f_length, length_new, t0, tRef, d_X_ASDinv, d_Y_ASDinv, d_Z_ASDinv);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void PhenomHM::Likelihood (int like_length, double *like_out_){

     double d_h = 0.0;
     double h_h = 0.0;
     char * status;
     double res;
     cuDoubleComplex result;
     for (int mode_i=0; mode_i<num_modes; mode_i++){
         stat = cublasZdotc(handle, like_length,
                 &d_X[mode_i*like_length], 1,
                 d_data_stream, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);

         stat = cublasZdotc(handle, like_length,
                 &d_Y[mode_i*like_length], 1,
                 d_data_stream, 1,
                 &result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();

          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         d_h += cuCreal(result);

         stat = cublasZdotc(handle, like_length,
                 &d_Z[mode_i*like_length], 1,
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
     stat = cublasDznrm2(handle, num_modes*like_length,
             d_X, 1, &res);
     status = _cudaGetErrorEnum(stat);
      cudaDeviceSynchronize();

      if (stat != CUBLAS_STATUS_SUCCESS) {
              exit(0);
          }
        h_h += res;

      // d_Y d_Y for h_h
      stat = cublasDznrm2(handle, num_modes*like_length,
              d_Y, 1, &res);
      status = _cudaGetErrorEnum(stat);
       cudaDeviceSynchronize();

       if (stat != CUBLAS_STATUS_SUCCESS) {
               exit(0);
           }
         h_h += res;

       // d_Z d_Z for h_h
       stat = cublasDznrm2(handle, num_modes*like_length,
               d_Z, 1, &res);
       status = _cudaGetErrorEnum(stat);
        cudaDeviceSynchronize();

        if (stat != CUBLAS_STATUS_SUCCESS) {
                exit(0);
            }
     h_h += res;

     like_out_[0] = d_h;
     like_out_[1] = h_h;
}


void PhenomHM::GetWaveform (std::complex<double>* X_, std::complex<double>* Y_, std::complex<double>* Z_) {
  assert(to_gpu == 1);
  gpuErrchk(cudaMemcpy(X_, d_X, data_stream_length*num_modes*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Y_, d_Y, data_stream_length*num_modes*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(Z_, d_Z, data_stream_length*num_modes*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
}

PhenomHM::~PhenomHM() {
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete[] pDPreComp_all_trans;
  delete[] q_all_trans;
  cpu_destroy_modes(mode_vals);
  delete[] H;

  cudaFree(d_freqs);
  cudaFree(d_data_stream);
  gpu_destroy_modes(d_mode_vals);
  cudaFree(d_pHM_trans);
  cudaFree(d_pAmp_trans);
  cudaFree(d_amp_prefactors_trans);
  cudaFree(d_pDPreComp_all_trans);
  cudaFree(d_q_all_trans);
  cudaFree(d_cShift);
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_Z);
  cudaFree(d_X_ASDinv);
  cudaFree(d_Y_ASDinv);
  cudaFree(d_Z_ASDinv);
  cublasDestroy(handle);
  cudaFree(d_B);
}
