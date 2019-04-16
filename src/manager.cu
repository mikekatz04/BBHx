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


using namespace std;

GPUPhenomHM::GPUPhenomHM (double *freqs_,
    int f_length_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    int to_gpu_){

    freqs = freqs_;
    f_length = f_length_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    to_gpu = to_gpu_;

    f_length = f_length_;

    cudaError_t err;

    // DECLARE ALL THE  NECESSARY STRUCTS

    freqs_geom_trans = new double[f_length];

    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

    factorp_trans = new std::complex<double>[num_modes];
    factorc_trans = new std::complex<double>[num_modes];


  if ((to_gpu == 0) || (to_gpu == 2)){
      printf("cpu\n");
      hptilde = new std::complex<double>[num_modes*f_length];
      hctilde = new std::complex<double>[num_modes*f_length];
  }
  if ((to_gpu == 1) || (to_gpu == 2)){

      printf("was here\n");

      size_t freqs_size = f_length*sizeof(double);
      cudaMalloc(&d_freqs_geom, freqs_size);

      size_t mode_array_size = num_modes*sizeof(unsigned int);
      cudaMalloc(&d_l_vals, mode_array_size);
      cudaMalloc(&d_m_vals, mode_array_size);
      cudaMemcpy(d_l_vals, l_vals, mode_array_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_m_vals, m_vals, mode_array_size, cudaMemcpyHostToDevice);

      size_t h_size = num_modes*f_length*sizeof(cuDoubleComplex);
      cudaMalloc(&d_hptilde, h_size);
      cudaMalloc(&d_hctilde, h_size);


      // DECLARE ALL THE  NECESSARY STRUCTS
      cudaMalloc(&d_pHM_trans, sizeof(PhenomHMStorage));

      cudaMalloc(&d_pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients));

      cudaMalloc(&d_amp_prefactors_trans, sizeof(AmpInsPrefactors));

      cudaMalloc(&d_pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp));

      err = cudaMalloc((void**) &d_q_all_trans, num_modes*sizeof(HMPhasePreComp));
      assert(err == 0);

      size_t complex_factor_size = num_modes*sizeof(cuDoubleComplex);
      err = cudaMalloc(&d_factorp_trans, complex_factor_size);
      assert(err == 0);
      err = cudaMalloc(&d_factorc_trans, complex_factor_size);
      assert(err == 0);

      double cShift[7] = {0.0,
                           PI_2 /* i shift */,
                           0.0,
                           -PI_2 /* -i shift */,
                           PI /* 1 shift */,
                           PI_2 /* -1 shift */,
                           0.0};

      err = cudaMalloc(&d_cShift, 7*sizeof(double));
      assert(err == 0);
      err = cudaMemcpy(d_cShift, &cShift, 7*sizeof(double), cudaMemcpyHostToDevice);
      assert(err == 0);

      // for likelihood
      // --------------
      cudaMallocHost((cuDoubleComplex**) &result, sizeof(cuDoubleComplex));

      stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }
      // ----------------

      NUM_THREADS = 256;
      num_blocks = std::ceil((f_length + NUM_THREADS -1)/NUM_THREADS);
      dim3 gridDim(num_modes, num_blocks);
      printf("blocks %d\n", num_blocks);
      this->gridDim = gridDim;
  }




  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;
}


void GPUPhenomHM::gpu_gen_PhenomHM(
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    assert((to_gpu == 1) || (to_gpu == 2));

    GPUPhenomHM::cpu_gen_PhenomHM(
        m1_, //solar masses
        m2_, //solar masses
        chi1z_,
        chi2z_,
        distance_,
        inclination_,
        phiRef_,
        deltaF_,
        f_ref_);


    // Initialize inputs

    cudaError_t err;

    err = cudaMemcpy(d_freqs_geom, freqs_geom_trans, f_length*sizeof(double), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMemcpy(d_pHM_trans, pHM_trans, sizeof(PhenomHMStorage), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMemcpy(d_pAmp_trans, pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMemcpy(d_amp_prefactors_trans, amp_prefactors_trans, sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMemcpy(d_pDPreComp_all_trans, pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMemcpy(d_q_all_trans, q_all_trans, num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMemcpy(d_factorp_trans, factorp_trans, num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    assert(err == 0);
    err = cudaMemcpy(d_factorc_trans, factorc_trans, num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    assert(err == 0);


    /* main: evaluate model at given frequencies */

    kernel_calculate_all_modes<<<gridDim, NUM_THREADS>>>(d_hptilde,
          d_hctilde,
          d_l_vals,
          d_m_vals,
          d_pHM_trans,
          d_freqs_geom,
          d_pAmp_trans,
          d_amp_prefactors_trans,
          d_pDPreComp_all_trans,
          d_q_all_trans,
          amp0,
          d_factorp_trans,
          d_factorc_trans,
          num_modes,
          f_length,
          t0,
          phi0,
          d_cShift
      );
     cudaDeviceSynchronize();
     err = cudaGetLastError();
     assert(err == 0);
}


void GPUPhenomHM::cpu_gen_PhenomHM(
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    inclination = inclination_;
    phiRef = phiRef_;
    deltaF = deltaF_;
    f_ref = f_ref_;

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        hptilde,
        hctilde,
        freqs,
        freqs_geom_trans,
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
        l_vals,
        m_vals,
        num_modes,
        to_gpu,
        pHM_trans,
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        factorp_trans,
        factorc_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in
}

double GPUPhenomHM::Likelihood (){

    stat = cublasZdotc(handle, f_length*num_modes,
            d_hptilde, 1,
            d_hptilde, 1,
            result);

    if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("CUBLAS initialization failed\n");
            return EXIT_FAILURE;
        }
    return cuCreal(result[0]);
}

void GPUPhenomHM::Get_Waveform (std::complex<double>* hptilde_, std::complex<double>* hctilde_) {

assert ((to_gpu == 0) || (to_gpu == 2));
 memcpy(hptilde_, hptilde, num_modes*f_length*sizeof(std::complex<double>));
 memcpy(hctilde_, hctilde, num_modes*f_length*sizeof(std::complex<double>));

}

void GPUPhenomHM::gpu_Get_Waveform (std::complex<double>* hptilde_, std::complex<double>* hctilde_) {
  assert((to_gpu == 1) || (to_gpu == 2));
    cudaError_t err;
     err = cudaMemcpy(hptilde_, d_hptilde, num_modes*f_length*sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
     assert(err == 0);
     err = cudaMemcpy(hctilde_, d_hctilde, num_modes*f_length*sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
     assert(err == 0);
}

GPUPhenomHM::~GPUPhenomHM() {
  delete freqs_geom_trans;
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete pDPreComp_all_trans;
  delete q_all_trans;
  delete factorp_trans;
  delete factorc_trans;

  if ((to_gpu ==0) || (to_gpu == 2)){
      delete hptilde;
      delete hctilde;
  }
  if ((to_gpu == 1) || (to_gpu == 2)){
      cudaFree(d_freqs_geom);
      cudaFree(d_l_vals);
      cudaFree(d_m_vals);
      cudaFree(d_pHM_trans);
      cudaFree(d_pAmp_trans);
      cudaFree(d_amp_prefactors_trans);
      cudaFree(d_pDPreComp_all_trans);
      cudaFree(d_q_all_trans);
      cudaFree(d_factorp_trans);
      cudaFree(d_factorc_trans);
      cudaFree(d_hptilde);
      cudaFree(d_hctilde);
      cudaFree(d_cShift);
      cudaFree(result);
      cublasDestroy(handle);
  }
}
