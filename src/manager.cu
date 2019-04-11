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
using namespace std;

GPUAdder::GPUPhenomHM (
    double *freqs,
    double m1, //solar masses
    double m2, //solar masses
    double chi1z,
    double chi2z,
    double distance,
    double inclination,
    double phiRef,
    double deltaF,
    double f_ref,
    double *l_vals,
    double *m_vals,
    int to_gpu){


    PhenomHMStorage *pHM_trans = (PhenomHMStorage*)malloc(sizeof(PhenomHMStorage));
    IMRPhenomDAmplitudeCoefficients *pAmp_trans = NULL;
    AmpInsPrefactors *amp_prefactors_trans = NULL;
    PhenDAmpAndPhasePreComp *pDPreComp_all_trans = NULL;
    HMPhasePreComp *q_all_trans = NULL;
    double complex *factorp_trans = NULL;
    double complex *factorc_trans = NULL;
    double t0;
    double phi0;
    double amp0;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        hptilde,
        hctilde,

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
        &pAmp_trans,
        &amp_prefactors_trans,
        &pDPreComp_all_trans,
        &q_all_trans,
        &factorp_trans,
        &factorc_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in IMRPhenomHM.");

    free(pHM_trans);
    free(pAmp_trans);
    free(amp_prefactors_trans);
    free(pDPreComp_all_trans);
    free(q_all_trans);
    free(factorp_trans);
    free(factorc_trans);int* array_host_, int length_) {
  array_host = array_host_;
  length = length_;
  double_errthing(array_host, length);
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

void GPUAdder::increment() {
  kernel_add_one<<<64, 64>>>(array_device, length, d_x);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void GPUAdder::retreive() {
  int size = length * sizeof(int);
  int sizex = sizeof(StructTest);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, sizex, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0) { cout << err << endl; assert(0); }
  cout << x->a;
}


void GPUAdder::retreive_to (int* array_host_, int length_) {
  assert(length == length_);
  int size = length * sizeof(int);
  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

GPUAdder::~GPUAdder() {
  cudaFree(array_device);
  cudaFree(d_x);
  free(x);
}
