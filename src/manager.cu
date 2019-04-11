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
//#include "tester.hh"
using namespace std;

GPUAdder::GPUPhenomHM (int* array_host_, int length_) {
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
