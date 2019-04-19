#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"
#include "cublas_v2.h"
#include "assert.h"
#include "tester.hh"
#include "PhenomHM.h"
#include "interpolate.hh"

class GPUPhenomHM {
  // pointer to the GPU memory where the array is stored
  int max_length;
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

  double *d_freqs;
  unsigned int *d_l_vals;
  unsigned int *d_m_vals;
  PhenomHMStorage *d_pHM_trans;
  IMRPhenomDAmplitudeCoefficients *d_pAmp_trans;
  AmpInsPrefactors *d_amp_prefactors_trans;
  PhenDAmpAndPhasePreComp *d_pDPreComp_all_trans;
  HMPhasePreComp *d_q_all_trans;
  double *d_cShift;

  dim3 gridDim;
  int NUM_THREADS;
  int num_blocks;

  cublasHandle_t handle;
  cublasStatus_t stat;
  cuDoubleComplex *result;

  // Interpolate related stuff
  int to_interp;
  ModeContainer *out_mode_vals;
  ModeContainer *d_out_mode_vals;
  Interpolate interp;
  int max_interp_length;
  double *interp_freqs;
  double *d_interp_freqs;

  ModeContainer *mode_vals;
  ModeContainer *d_mode_vals;

  int *h_indices;
  int *d_indices;


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
      int num_modes_,
      int to_gpu_,
      int to_interp_); // constructor (copies to GPU)

  ~GPUPhenomHM(); // destructor

  void add_interp(int max_interp_length_);

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

    void gpu_gen_PhenomHM(double *freqs_, int f_length_,
        double m1_, //solar masses
        double m2_, //solar masses
        double chi1z_,
        double chi2z_,
        double distance_,
        double inclination_,
        double phiRef_,
        double deltaF_,
        double f_ref_);

  void interp_wave(double f_min, double dF, int length_new);

  double Likelihood ();
  //gets results back from the gpu, putting them in the supplied memory location
  void Get_Waveform (int mode_i, double* amp_, double* phase_);
  void gpu_Get_Waveform (int mode_i, double* amp_, double* phase_);


};


#endif //__MANAGER_H__
