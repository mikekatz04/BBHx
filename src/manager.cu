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
    double *freqs,
    int f_length_,
    double m1, //solar masses
    double m2, //solar masses
    double chi1z,
    double chi2z,
    double distance,
    double inclination,
    double phiRef,
    double deltaF,
    double f_ref,
    unsigned int *l_vals,
    unsigned int *m_vals,
    int num_modes,
    int to_gpu){

    f_length = f_length_;

    //COMPLEX2dArray *hptilde = CreateCOMPLEX2dArray(f_length, num_modes);
    //COMPLEX2dArray *hctilde = CreateCOMPLEX2dArray(f_length, num_modes);
    // DECLARE ALL THE  NECESSARY STRUCTS FOR THE GPU
    PhenomHMStorage *pHM_trans = (PhenomHMStorage *)malloc(sizeof(PhenomHMStorage));
    IMRPhenomDAmplitudeCoefficients *pAmp_trans = (IMRPhenomDAmplitudeCoefficients*)malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
    AmpInsPrefactors *amp_prefactors_trans = (AmpInsPrefactors*)malloc(sizeof(AmpInsPrefactors));
    PhenDAmpAndPhasePreComp *pDPreComp_all_trans = (PhenDAmpAndPhasePreComp*)malloc(num_modes*sizeof(PhenDAmpAndPhasePreComp));
    HMPhasePreComp *q_all_trans = (HMPhasePreComp*)malloc(num_modes*sizeof(HMPhasePreComp));
    double complex *factorp_trans = (double complex*)malloc(num_modes*sizeof(double complex));
    double complex *factorc_trans = (double complex*)malloc(num_modes*sizeof(double complex));
    double t0;
    double phi0;
    double amp0;

    /* main: evaluate model at given frequencies */
    /*retcode = 0;
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
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        factorp_trans,
        factorc_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in IMRPhenomHM.");*/

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

GPUPhenomHM::~GPUPhenomHM() {
  cudaFree(array_device);
  cudaFree(d_x);
  free(pHM_trans);
  free(pAmp_trans);
  free(amp_prefactors_trans);
  free(pDPreComp_all_trans);
  free(q_all_trans);
  free(factorp_trans);
  free(factorc_trans);
  free(x);
  //DestroyCOMPLEX2dArray(hptilde);
  //DestroyCOMPLEX2dArray(hctilde);
}
