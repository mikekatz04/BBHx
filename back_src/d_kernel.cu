#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_complex.hpp"

#include "globalPhenomHM.h"
#include "kernel_response.hh"
#include "IMRPhenomD.h"
#include "IMRPhenomD_internals.h"
#include "interpolate.hh"

#ifdef __CUDACC__
#else
#include "omp.h"
#endif




#ifdef __GLOBAL_FIT__
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
#else
CUDA_CALLABLE_MEMBER
#endif // __GLOBAL_FIT__
  void calculate_each_mode_PhenomD(int i, ModeContainer mode_val,
       double freq_geom,
       PhenDAmpAndPhasePreComp pDPreComp,
       double amp0, double t0, double phi0, double Mf_ref, agcmplx *H, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int data_length,
			     double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers,
			     double inc, double lam, double beta, double psi, double tRef_wave_frame, double tRef_sampling_frame, double merger_freq,
			     int mode_index, double f, int walker_i, double M_tot_sec, double t_obs_end, double start_time,
					 agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, double *channel1_ASDinv,
	 		    double *channel2_ASDinv, double *channel3_ASDinv){
			     // TDItag == 1 is XYZ, TDItag == 2 is AET
			     double phasetimeshift;
			     double phi_up, phi;

			     double t, t_wave_frame, t_sampling_frame, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3, f_last, Shift, t_merger, dphidf, dphidf_merger;
			     int old_ind_below;

			           double Mf = f*M_tot_sec;
								 double f_ref = Mf_ref/M_tot_sec;

			           PNPhasingSeries pn = pDPreComp.pn;
			           IMRPhenomDPhaseCoefficients pPhi = pDPreComp.pPhi;

			           dphidf = -1.0*DPhiInsAnsatzInt(Mf, &pPhi, &pn)*M_tot_sec;

			             t_wave_frame = 1./(2.0*PI)*dphidf + tRef_wave_frame;
			             t_sampling_frame = 1./(2.0*PI)*dphidf + tRef_sampling_frame;

			             if ((start_time - t_sampling_frame > t_obs_end) || (f < f_ref)){
                     agcmplx trans_complex1(0.0, 0.0);
                     agcmplx trans_complex2(0.0, 0.0);
                     agcmplx trans_complex3(0.0, 0.0);

                     fill_templates(channel1_out, channel2_out, channel3_out, i, walker_i, data_length,
                                    trans_complex1, trans_complex2, trans_complex3,
                                     channel1_ASDinv, channel2_ASDinv, channel3_ASDinv);
			               return;
			             }
			             t_sampling_frame = start_time - t_sampling_frame;
			             t_wave_frame = start_time - t_wave_frame;
			             //printf("%d, %e, %e, %e, %e, %e, %e\n", i, f, start_time, p, start_time - t_sampling_frame, t_obs_end, dphidf);

			             // adjust phase values stored in mode vals to reflect the tRef shift
			             // in phenD do not need to do after response because the dphidf is determined analytically

			             d_transferL_holder transferL = d_JustLISAFDresponseTDI(&H[mode_index*9], f, t_wave_frame, lam, beta, t0, TDItag, order_fresnel_stencil);

           double phase_term1, phase_term2;
           int status_in_for;
           UsefulPowers powers_of_f;

					 PhiInsPrefactors phi_prefactors = pDPreComp.phi_prefactors;
					 IMRPhenomDAmplitudeCoefficients pAmp = pDPreComp.pAmp;
					 AmpInsPrefactors amp_prefactors = pDPreComp.amp_prefactors;

           int retcode = 0;

           Mf = freq_geom;

           status_in_for = init_useful_powers(&powers_of_f, Mf);
                /*if (PD_SUCCESS != status_in_for)
                {
                  //printf("init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
                  retcode = status_in_for;
                  //exit(0);
                }
                else
                {*/

					double f_seven_sixths = Mf * (&powers_of_f)->sixth;
					double AmpPreFac = (&amp_prefactors)->amp0 / f_seven_sixths;

					double AmpIns = AmpPreFac * AmpInsAnsatz(Mf, &powers_of_f, &amp_prefactors);

               double amp = AmpIns*amp0;

              /* Add complex phase shift depending on 'm' mode */
							double PhiIns = PhiInsAnsatzInt(Mf, &powers_of_f, &phi_prefactors, &pPhi, &pn);

              phase_term1 = 0.0; // - t0 * (Mf - Mf_ref); // phenomD t0 is zero at start of signal at t_ref=0.0
              phase_term2 = PhiIns - (2 * phi0);

              double phase = (phase_term1 + phase_term2)+ 2.0*PI*f*tRef_wave_frame;

							double phaseRdelay  =  transferL.phaseRdelay;
              double phaseShift = 0.0;

						  agcmplx ampphasefactor = get_ampphasefactor(amp, phase, phaseRdelay, phaseShift);


						  agcmplx trans_complex1 = combine_information(ampphasefactor, transferL.transferL1.real(), transferL.transferL1.imag()); //TODO may be faster to load as complex number with 0.0 for imaginary part
							agcmplx trans_complex2 = combine_information(ampphasefactor, transferL.transferL2.real(), transferL.transferL2.imag());
							agcmplx trans_complex3 = combine_information(ampphasefactor, transferL.transferL3.real(), transferL.transferL3.imag());

              fill_templates(channel1_out, channel2_out, channel3_out, i, walker_i, data_length,
                             trans_complex1, trans_complex2, trans_complex3,
                              channel1_ASDinv, channel2_ASDinv, channel3_ASDinv);
  }


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
		    double *channel2_ASDinv, double *channel3_ASDinv	){
						// TDItag == 1 is XYZ, TDItag == 2 is AET
						double inc, lam, beta, psi, tRef_wave_frame, tRef_sampling_frame, merger_freq, start_time;
						PhenDAmpAndPhasePreComp pDPreComp;
						int mode_index, freq_ind;
						double freq_geom;
						double Mf_ref;

						double f;

						double f_max_limit = 0.025;

						for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
								 walker_i < num_walkers;
								 walker_i += blockDim.z * gridDim.z){

								inc = inc_arr[walker_i];
								lam = lam_arr[walker_i];
								beta = beta_arr[walker_i];
								psi = psi_arr[walker_i];
								tRef_wave_frame = tRef_wave_frame_arr[walker_i];
								tRef_sampling_frame = tRef_sampling_frame_arr[walker_i];
								merger_freq = merger_freq_arr[walker_i];
								pDPreComp = pDPreComp_arr[walker_i];
								start_time = start_time_arr[walker_i];
								Mf_ref = f_ref[walker_i]*M_tot_sec[walker_i];
								double phi0_t = phi0[walker_i];
								double f_ref_t = f_ref[walker_i];

								double amp0_t = amp0[walker_i];

								double t0_t = t0[walker_i];

								double M_tot_sec_t = M_tot_sec[walker_i];

								ModeContainer mode_val = mode_vals[walker_i];

											mode_index = walker_i;


						for (int i = blockIdx.x * blockDim.x + threadIdx.x;
								 i < num_points;
								 i += blockDim.x * gridDim.x){

								 f = frqs[i];
								 if (f > f_max_limit) return;

								 if (f < f_ref_t) continue;


           freq_geom = freqs[i]*M_tot_sec_t;

           calculate_each_mode_PhenomD(i, mode_val,
				        freq_geom,
				        pDPreComp,
				        amp0_t, t0_t, phi0_t, Mf_ref, H, old_freqs, d_log10f, l_vals, m_vals, num_modes, num_points,
				 			     merger_freq_arr, TDItag, order_fresnel_stencil, num_walkers,
				 			     inc, lam, beta, psi, tRef_wave_frame, tRef_sampling_frame, merger_freq,
				 			     mode_index, f, walker_i, M_tot_sec_t, t_obs_end, start_time,
				 					 channel1_out, channel2_out, channel3_out, channel1_ASDinv,
				 	 		    channel2_ASDinv, channel3_ASDinv);

        }
    }
	}
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
		    double *channel2_ASDinv, double *channel3_ASDinv	){
						// TDItag == 1 is XYZ, TDItag == 2 is AET
						double inc, lam, beta, psi, tRef_wave_frame, tRef_sampling_frame, merger_freq, start_time;
						PhenDAmpAndPhasePreComp pDPreComp;
						int mode_index, freq_ind;
						double freq_geom;
						double Mf_ref;

						double f;

						double f_max_limit = 0.025;

            #pragma omp for collapse(2)
						for (int walker_i = 0;
								 walker_i < num_walkers;
								 walker_i += 1){

						for (int i = 0;
								 i < num_points;
								 i += 1){

								 f = frqs[i];

                 inc = inc_arr[walker_i];
                 lam = lam_arr[walker_i];
                 beta = beta_arr[walker_i];
                 psi = psi_arr[walker_i];
                 tRef_wave_frame = tRef_wave_frame_arr[walker_i];
                 tRef_sampling_frame = tRef_sampling_frame_arr[walker_i];
                 merger_freq = merger_freq_arr[walker_i];
                 pDPreComp = pDPreComp_arr[walker_i];
                 start_time = start_time_arr[walker_i];
                 Mf_ref = f_ref[walker_i]*M_tot_sec[walker_i];
                 double phi0_t = phi0[walker_i];
                 double f_ref_t = f_ref[walker_i];

                 double amp0_t = amp0[walker_i];

                 double t0_t = t0[walker_i];

                 double M_tot_sec_t = M_tot_sec[walker_i];

                 ModeContainer mode_val = mode_vals[walker_i];

                       mode_index = walker_i;

             if (f > f_max_limit) continue;

             if (f < f_ref_t) continue;


             freq_geom = freqs[i]*M_tot_sec_t;

             calculate_each_mode_PhenomD(i, mode_val,
  				        freq_geom,
  				        pDPreComp,
  				        amp0_t, t0_t, phi0_t, Mf_ref, H, old_freqs, d_log10f, l_vals, m_vals, num_modes, num_points,
  				 			     merger_freq_arr, TDItag, order_fresnel_stencil, num_walkers,
  				 			     inc, lam, beta, psi, tRef_wave_frame, tRef_sampling_frame, merger_freq,
  				 			     mode_index, f, walker_i, M_tot_sec_t, t_obs_end, start_time,
  				 					 channel1_out, channel2_out, channel3_out, channel1_ASDinv,
  				 	 		    channel2_ASDinv, channel3_ASDinv);

        }
    }
	}
	#endif
