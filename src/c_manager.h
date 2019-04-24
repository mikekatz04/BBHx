#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "globalPhenomHM.h"
#include <complex>
#include "assert.h"
#include "PhenomHM.h"

class GPUPhenomHM {
  // pointer to the GPU memory where the array is stored
  int max_length;
  int data_stream_length;
  double *freqs;
  int f_length;
  double m1; //solar masses
  double m2; //solar masses
  double chi1z;
  double chi2z;
  double distance;
  double inclination;
  double phiRef;
  double deltaF;
  double f_ref;
  unsigned int *l_vals;
  unsigned int *m_vals;
  int num_modes;
  int to_gpu;
  PhenomHMStorage *pHM_trans;
  IMRPhenomDAmplitudeCoefficients *pAmp_trans;
  AmpInsPrefactors *amp_prefactors_trans;
  PhenDAmpAndPhasePreComp *pDPreComp_all_trans;
  HMPhasePreComp *q_all_trans;
  double t0;
  double phi0;
  double amp0;
  int retcode;
  double m1_SI;
  double m2_SI;

  // Interpolate related stuff
  int to_interp;
  ModeContainer *out_mode_vals;
  int max_interp_length;
  double *interp_freqs;

  ModeContainer *mode_vals;

  int *h_indices;

  std::complex<double> *data_stream;

public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  GPUPhenomHM(int max_length_,
      unsigned int *l_vals_,
      unsigned int *m_vals_,
      int num_modes_); // constructor (copies to GPU)

  ~GPUPhenomHM(); // destructor

  void cpu_gen_PhenomHM(double *freqs_, int f_length_,
      double m1_, //solar masses
      double m2_, //solar masses
      double chi1z_,
      double chi2z_,
      double distance_,
      double inclination_,
      double phiRef_,
      double deltaF_,
      double f_ref_);

  //gets results back from the gpu, putting them in the supplied memory location
  void Get_Waveform (int mode_i, double* amp_, double* phase_);

};


#endif //__MANAGER_H__
