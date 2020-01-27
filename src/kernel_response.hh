#ifndef _KERNEL_RESPONSE_
#define _KERNEL_RESPONSE_

#include "globalPhenomHM.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "fdresponse.h"


#ifdef __CUDACC__
CUDA_KERNEL
void kernel_JustLISAFDresponseTDI_wrap(
#else
void cpu_JustLISAFDresponseTDI_wrap(
#endif
                ModeContainer *mode_vals, agcmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double *inc_arr, double *lam_arr, double *beta_arr, double *psi_arr, double *phi0_arr, double *t0_arr, double *tRef_wave_frame_arr, double *tRef_sampling_frame_arr,
    double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers
  );

#endif //_KERNEL_RESPONSE_
