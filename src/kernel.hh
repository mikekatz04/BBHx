#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_complex.hpp"
#include "globalPhenomHM.h"

#ifdef __CUDACC__
CUDA_KERNEL
void kernel_calculate_all_modes_wrap(
        ModeContainer *mode_vals,
        PhenomHMStorage *pHM,
        double *freqs,
        double *M_tot_sec,
        IMRPhenomDAmplitudeCoefficients *pAmp,
        AmpInsPrefactors *amp_prefactors,
        PhenDAmpAndPhasePreComp *pDPreComp_all,
        HMPhasePreComp *q_all,
        double *amp0,
        int num_modes,
        double *t0,
        double *phi0,
        double *cshift,
      int nwalkers,
      int length,
      int walker_i
    );
#else
void cpu_calculate_all_modes_wrap(

      ModeContainer *mode_vals,
      PhenomHMStorage *pHM,
      double *freqs,
      double *M_tot_sec,
      IMRPhenomDAmplitudeCoefficients *pAmp,
      AmpInsPrefactors *amp_prefactors,
      PhenDAmpAndPhasePreComp *pDPreComp_all,
      HMPhasePreComp *q_all,
      double *amp0,
      int num_modes,
      double *t0,
      double *phi0,
      double *cshift,
    int nwalkers,
    int length
  );
#endif
#endif // _KERNEL_H_
