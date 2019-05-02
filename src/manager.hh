#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"
#include "cublas_v2.h"
#include "PhenomHM.h"
#include "interpolate.hh"

class PhenomHM {
  // pointer to the GPU memory where the array is stored
  int current_status;
  int max_length_init;
  int data_stream_length;
  double *freqs;
  int current_length;
  double m1; //solar masses
  double m2; //solar masses
  double chi1z;
  double chi2z;
  double distance;
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
  cmplx *H;

  double *d_freqs;
  double *d_time_freq_corr;
  unsigned int *d_l_vals;
  unsigned int *d_m_vals;
  PhenomHMStorage *d_pHM_trans;
  IMRPhenomDAmplitudeCoefficients *d_pAmp_trans;
  AmpInsPrefactors *d_amp_prefactors_trans;
  PhenDAmpAndPhasePreComp *d_pDPreComp_all_trans;
  HMPhasePreComp *d_q_all_trans;
  double *d_cShift;
  cuDoubleComplex *d_H;

  dim3 gridDim;
  int NUM_THREADS;
  int num_blocks;

  cublasHandle_t handle;
  cublasStatus_t stat;

  // Interpolate related stuff
  Interpolate interp;
  double *d_B;

  ModeContainer *mode_vals;
  ModeContainer *d_mode_vals;

  double *data_freqs;
  double *d_data_freqs;
  cmplx *data_stream;

  double *channel1_ASDinv;
  double *channel2_ASDinv;
  double *channel3_ASDinv;

  cuDoubleComplex *d_data_stream;
  cuDoubleComplex *d_template_channel1;
  cuDoubleComplex *d_template_channel2;
  cuDoubleComplex *d_template_channel3;
  double *d_channel1_ASDinv;
  double *d_channel2_ASDinv;
  double *d_channel3_ASDinv;

  double inc;
  double lam;
  double beta;
  double psi;
  double t0_epoch;
  double tRef;
  int TDItag;
  double merger_freq;


public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  PhenomHM(int max_length_init_,
      unsigned int *l_vals_,
      unsigned int *m_vals_,
      int num_modes_,
      double *data_freqs_,
      cmplx *data_stream_, int data_stream_length_, double *X_ASDinv_, double *Y_ASDinv_, double *Z_ASDinv_, int TDItag); // constructor (copies to GPU)

  ~PhenomHM(); // destructor

    void gen_amp_phase(double *freqs_, int current_length_,
        double m1_, //solar masses
        double m2_, //solar masses
        double chi1z_,
        double chi2z_,
        double distance_,
        double phiRef_,
        double f_ref_);

    void gen_amp_phase_prep(double *freqs_, int current_length_,
            double m1_, //solar masses
            double m2_, //solar masses
            double chi1z_,
            double chi2z_,
            double distance_,
            double phiRef_,
            double f_ref_);

  void setup_interp_wave();

  void LISAresponseFD(double inc, double lam, double beta, double psi, double t0_epoch, double tRef, double merger_freq);

  void setup_interp_response();

  void perform_interp();

  void Likelihood (double *like_out_);

  void GetTDI (cmplx* X_, cmplx* Y_, cmplx* Z_);

  void GetAmpPhase(double* amp_, double* phase_);
};


static char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#endif //__MANAGER_H__
