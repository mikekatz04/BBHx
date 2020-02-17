#ifndef _CREATE_
#define _CREATE_

#include "globalPhenomHM.h"
#include "stdio.h"

class WalkerContainer {

    int walker_i;
    int num_modes;
    int init_length;

    int start_ind;
    int end_ind;

    double phi0, amp0, t0; // constants for phenomHM

    double m1, m2, chi1z, chi2z, distance;

    double phiRef, inc, lam, beta, psi, tRef_sampling_frame, tRef_wave_frame;

    ModeContainer *mode_vals;
    ModeContainer *h_mode_vals;

    PhenomHMStorage *pHM_trans;
    IMRPhenomDAmplitudeCoefficients *pAmp_trans;
    AmpInsPrefactors *amp_prefactors_trans;
    PhenDAmpAndPhasePreComp *pDPreComp_all_trans;
    HMPhasePreComp *q_all_trans;

  public:

    WalkerContainer();

    void fill_info(int walker_i, int num_modes, unsigned int *l_vals_, unsigned int *m_vals_, int init_length_);
    void remove_info();

    ~WalkerContainer();

};

void gpu_create_modes(ModeContainer *mode_vals, ModeContainer *h_mode_vals, int num_modes, int num_walkers, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp);
void gpu_destroy_modes(ModeContainer *d_mode_vals, ModeContainer *mode_vals);

/*
Function for gpu Error checking.
//*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#endif // _CREATE_
