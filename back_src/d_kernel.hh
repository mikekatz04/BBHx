#ifndef __D_KERNEL__
#define __D_KERNEL__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_complex.hpp"

#include "globalPhenomHM.h"
#include "kernel.hh"
#include "kernel_response.hh"


#ifdef __CUDACC__
  CUDA_KERNEL
  void kernel_calculate_all_modes_PhenomD(ModeContainer *mode_vals,
        double *freqs,
        double *M_tot_sec,
        PhenDAmpAndPhasePreComp *pDPreComp_all,
        double *amp0,
        double *t0,
        double *phi0,
        int length,
				double *f_ref,
				int nwalkers,
				agcmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double *inc_arr, double *lam_arr, double *beta_arr, double *psi_arr, double *phi0_arr, double *t0_arr, double *tRef_wave_frame_arr, double *tRef_sampling_frame_arr,
						double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers, double *M_tot_sec_arr, PhenDAmpAndPhasePreComp *pDPreComp_arr, double t_obs_end, double *start_time_arr,
				agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, double *channel1_ASDinv,
		    double *channel2_ASDinv, double *channel3_ASDinv	);
	#else

  void cpu_calculate_all_modes_PhenomD(ModeContainer *mode_vals,
        double *freqs,
        double *M_tot_sec,
        PhenDAmpAndPhasePreComp *pDPreComp_all,
        double *amp0,
        double *t0,
        double *phi0,
        int length,
        double *f_ref,
        int nwalkers,
        agcmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double *inc_arr, double *lam_arr, double *beta_arr, double *psi_arr, double *phi0_arr, double *t0_arr, double *tRef_wave_frame_arr, double *tRef_sampling_frame_arr,
            double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers, double *M_tot_sec_arr, PhenDAmpAndPhasePreComp *pDPreComp_arr, double t_obs_end, double *start_time_arr,
        agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, double *channel1_ASDinv,
        double *channel2_ASDinv, double *channel3_ASDinv	);

#endif

#endif // __D_KERNEL__
