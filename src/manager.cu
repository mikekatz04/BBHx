/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include "tester.hh"
#include "complex.h"


using namespace std;

GPUPhenomHM::GPUPhenomHM (int* array_host_, int length_,
    double *freqs_,
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

  if (!to_gpu){
      std::complex<double> *hptilde = new std::complex<double>[num_modes*f_length];
      std::complex<double> *hctilde = new std::complex<double>[num_modes*f_length];
      this->hptilde = hptilde;
      this->hctilde = hctilde;
  }

      // DECLARE ALL THE  NECESSARY STRUCTS

  PhenomHMStorage *pHM_trans = new PhenomHMStorage;
  this->pHM_trans = pHM_trans;

  IMRPhenomDAmplitudeCoefficients *pAmp_trans = new IMRPhenomDAmplitudeCoefficients;
  this->pAmp_trans = pAmp_trans;

  AmpInsPrefactors *amp_prefactors_trans = new AmpInsPrefactors;
  this->amp_prefactors_trans = amp_prefactors_trans;

  PhenDAmpAndPhasePreComp *pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];
  this->pDPreComp_all_trans = pDPreComp_all_trans;

  HMPhasePreComp *q_all_trans = new HMPhasePreComp[num_modes];
  this->q_all_trans = q_all_trans;

  std::complex<double> *factorp_trans = new std::complex<double>[num_modes];
  this->factorp_trans = factorp_trans;

  std::complex<double> *factorc_trans = new std::complex<double>[num_modes];
  this->factorc_trans = factorc_trans;

  //double t0_;
  this->t0 = 0.0;

  //double phi0_;
  this->phi0 = 0.0;

  //double amp0_;
  this->amp0 = 0.0;

  int retcode;

  array_host = array_host_;
  length = length_;
  int size = length * sizeof(int);
  cudaError_t err = cudaMalloc((void**) &array_device, size);
  assert(err == 0);
  err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
  assert(err == 0);

  int sizex = sizeof(StructTest);
  x = (StructTest*) malloc(sizex);
  x->a = 10;

  err = cudaMalloc((void**) &d_x, sizex);
  assert(err == 0);
  err = cudaMemcpy(d_x, x, sizex, cudaMemcpyHostToDevice);
  assert(err == 0);

}

void GPUPhenomHM::increment() {
  kernel_add_one<<<64, 64>>>(array_device, length, d_x);
  cudaError_t err = cudaGetLastError();
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
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in IMRPhenomHM.");
    /*int i, j;
    printf("f_length %d\n\n", f_length);
    double check;
    for (i=0; i<num_modes; i++){
        for (j=0; j<f_length; j++){
            check = std::real(hptilde[i*f_length + j]);
            if (j % 100 == 0) printf("%e, %e, %e, %e, %e\n", freqs[j], std::real(hptilde[i*f_length + j]), std::imag(hptilde[i*f_length + j]), std::real(hctilde[i*f_length + j]), std::imag(hctilde[i*f_length + j]));
        }
    }
    //this->hptilde = hptilde;
    printf("\n\n\n\n\n\n\n");
     printf("\nhptilde %e\n\n", hptilde[0].real());*/

}

void GPUPhenomHM::retreive() {
  int size = length * sizeof(int);
  int sizex = sizeof(StructTest);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, sizex, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0) { cout << err << endl; assert(0); }
  cout << x->a;
}


void GPUPhenomHM::retreive_to (int* array_host_, int length_) {
  assert(length == length_);
  int size = length * sizeof(int);
  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void GPUPhenomHM::Get_Waveform (std::complex<double>* hptilde_, std::complex<double>* hctilde_) {
  //hptilde[10] = std::complex<double>(10.0, 9.0);
  //printf("%e\n", hptilde[0].real());
  //printf("%d %d\n", length_, f_length);
if (this->to_gpu == 0){
     memcpy(hptilde_, hptilde, num_modes*f_length*sizeof(std::complex<double>));
     memcpy(hctilde_, hctilde, num_modes*f_length*sizeof(std::complex<double>));
}
  //array_host_[0] = this->hptilde[0];
  //printf("%e\n", array_host_[0].real());
}

GPUPhenomHM::~GPUPhenomHM() {
  cudaFree(array_device);
  cudaFree(d_x);
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete pDPreComp_all_trans;
  delete q_all_trans;
  delete factorp_trans;
  delete factorc_trans;
  free(x);
  //free(freqs);
  delete hptilde;
  delete hctilde;
}
