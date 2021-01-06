/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.

 * This code is adjusted for usage in CUDA. Refer to PhenomHM.cpp for comments.


 *  Copyright (C) 2017 Sebastian Khan, Francesco Pannarale, Lionel London
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_complex.hpp"

#include "globalPhenomHM.h"
#include "kernel.hh"
#include "IMRPhenomD_internals.h"
#include "PhenomHM.h"
#include "IMRPhenomD.h"

#ifdef __CUDACC__
#else
#include "omp.h"
#endif


CUDA_CALLABLE_MEMBER
 void calculate_each_mode(int i, ModeContainer mode_val,
     unsigned int ell,
     unsigned int mm,
     PhenomHMStorage *pHM,
     double freq_geom,
     IMRPhenomDAmplitudeCoefficients *pAmp,
     AmpInsPrefactors *amp_prefactors,
     PhenDAmpAndPhasePreComp pDPreComp,
     HMPhasePreComp q,
     double amp0,
     double Rholm, double Taulm, double t0, double phi0, double *cshift,
    int walker_i, int mode_i){

         double freq_amp, Mf, beta_term1, beta, beta_term2, HMamp_term1, HMamp_term2;
         double Mf_wf, Mfr, tmpphaseC, phase_term1, phase_term2;
         double amp_i, phase_i;
         int status_in_for;
         UsefulPowers powers_of_f;
         //cuDoubleComplex J = make_cuDoubleComplex(0.0, 1.0);
         int retcode = 0;

          /* loop over only positive m is intentional. negative m added automatically */
          // generate amplitude
          // IMRPhenomHMAmplitude


        freq_amp = IMRPhenomHMFreqDomainMap(freq_geom, ell, mm, pHM, AmpFlagTrue);

            //status_in_for = PD_SUCCESS;
          /* Now generate the waveform */
              Mf = freq_amp; //freqs->data[i]; // geometric frequency

              status_in_for = init_useful_powers(&powers_of_f, Mf);
              /*if (PD_SUCCESS != status_in_for)
              {
                //printf("init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
                retcode = status_in_for;
                //exit(0);
              }
              else
              {*/
                amp_i = IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, amp_prefactors);
             // }


            beta_term1 = IMRPhenomHMOnePointFiveSpinPN(
                freq_geom,
                ell,
                mm,
                pHM->m1,
                pHM->m2,
                pHM->chi1z,
                pHM->chi2z);

            beta=0.0;
            beta_term2=0.0;
            HMamp_term1=1.0;
            HMamp_term2=1.0;
            //HACK to fix equal black hole case producing NaNs.
            //More elegant solution needed.
            if (beta_term1 == 0.){
                beta = 0.;
            } else {

                beta_term2 = IMRPhenomHMOnePointFiveSpinPN(2.0 * freq_geom / mm, ell, mm, pHM->m1, pHM->m2, pHM->chi1z, pHM->chi2z);
                beta = beta_term1 / beta_term2;

                /* LL: Apply steps #1 and #2 */
                HMamp_term1 = IMRPhenomHMOnePointFiveSpinPN(
                    freq_amp,
                    ell,
                    mm,
                    pHM->m1,
                    pHM->m2,
                    pHM->chi1z,
                    pHM->chi2z);
                HMamp_term2 = IMRPhenomHMOnePointFiveSpinPN(freq_amp, 2, 2, pHM->m1, pHM->m2, 0.0, 0.0);

            }

            //HMamp is computed here
            amp_i *= beta * HMamp_term1 / HMamp_term2;

            mode_val.amp[i] = amp_i*amp0;

        Mf_wf = 0.0;
        Mf = 0.0;
        Mfr = 0.0;
        tmpphaseC = 0.0;
        //for (unsigned int i = pHM->ind_min; i < pHM->ind_max; i++)
        //{
            /* Add complex phase shift depending on 'm' mode */
            phase_i = cshift[mm];
            Mf_wf = freq_geom;
            // This if ladder is in the mathematica function HMPhase. PhenomHMDev.nb

            if (!(Mf_wf > q.fi))
            { /* in mathematica -> IMRPhenDPhaseA */
                Mf = q.ai * Mf_wf + q.bi;
                phase_i += IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.ai;
            }
            else if (!(Mf_wf > q.fr))
            { /* in mathematica -> IMRPhenDPhaseB */
                Mf = q.am * Mf_wf + q.bm;
                phase_i += IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.am - q.PhDBconst + q.PhDBAterm;
            }
            else if ((Mf_wf > q.fr))
            { /* in mathematica -> IMRPhenDPhaseC */
                Mfr = q.am * q.fr + q.bm;
                tmpphaseC = IMRPhenomDPhase_OneFrequency(Mfr, pDPreComp, Rholm, Taulm) / q.am - q.PhDBconst + q.PhDBAterm;
                Mf = q.ar * Mf_wf + q.br;
                phase_i += IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.ar - q.PhDCconst + tmpphaseC;
            }
            else
            {
                //printf("ERROR - should not get here - in function IMRPhenomHMPhase");
                //ERROR(PD_EDOM, "error");
            }

        //}

            //phase_term1 = 0.0;
            //phase_term2 = 0.0;
            //Mf = 0.0;
            Mf = freq_geom;
            phase_term1 = - t0 * (Mf - pHM->Mf_ref);
            phase_term2 = phase_i - (mm * phi0);

            mode_val.phase[i] = (phase_term1 + phase_term2);


             /* # if __CUDA_ARCH__>=200
                  if (((i == 1184) || (i==1183)) && (walker_i == 20))
                  printf("%d, %d, %d, %.12e\n", walker_i, mode_i, i, mode_val.phase[i]);
              #endif //*/


//hlm = cuCmul(make_cuDoubleComplex(amp_i, 0.0), my_cexpf(cuCmul(make_cuDoubleComplex(0.0, -1.0), make_cuDoubleComplex(phase_term1 + phase_term2, 0))));

             //double complexFrequencySeries *hlm = XLALSphHarmFrequencySeriesGetMode(*hlms, ell, mm);
             /*if (!(hlm))
             {
                 hptilde[mode_i*length + i] = 0.0; //TODO check += here
                 hctilde[mode_i*length + i] = 0.0;
             }
             else
             {*/
                // hptilde[mode_i*length + i] = cuCmul(cuCmul(factorp, hlm), make_cuDoubleComplex(amp0, 0.0)); //TODO check += here
                // hctilde[mode_i*length + i] = cuCmul(cuCmul(factorc, hlm), make_cuDoubleComplex(amp0, 0.0));
            // }

             //IMRPhenomHMFDAddMode(*hptilde, *hctilde, hlm, inclination, 0., ell, mm, sym); /* The phase \Phi is set to 0 - assumes phiRef is defined as half the phase of the 22 mode h22 */

             //if (mode_i == 1)
             //    printf("%d, %d %e\n", ell, mm, (*hptilde)->data[mode_i*(*hptilde)->length + i]);
         //printf("(l, m): (%d, %d)\n", ell, mm);
}


CUDA_CALLABLE_MEMBER
void kernel_calculate_all_modes(ModeContainer *mode_vals,
      PhenomHMStorage *pHM,
      double *freqs,
      double *M_tot_sec,
      IMRPhenomDAmplitudeCoefficients *pAmp,
      AmpInsPrefactors *amp_prefactors,
      PhenDAmpAndPhasePreComp *pDPreComp_all,
      HMPhasePreComp *q_all,
      double *amp0,
      int mode_i,
      double *t0,
      double *phi0,
      double *cshift,
	  int walker_i,
	  int i,
		int num_modes,
		int length
        ){
      unsigned int mm, ell;
      double Rholm, Taulm;
      double freq_geom;
      /* if (mode_i >= num_modes) return;
       for (int i = blockIdx.y * blockDim.x + threadIdx.x;
          i < length;
          i += blockDim.x * gridDim.y)*/

      if ((i < (&pHM[walker_i])->ind_max) && (i >= (&pHM[walker_i])->ind_min))  // kernel setup should always make second part true
      {

         ell = mode_vals[walker_i*num_modes + mode_i].l;
         mm = mode_vals[walker_i*num_modes + mode_i].m;
         Rholm = (&pHM[walker_i])->Rholm[ell][mm];
         Taulm = (&pHM[walker_i])->Taulm[ell][mm];
         freq_geom = freqs[walker_i*length + i]*M_tot_sec[walker_i];

         calculate_each_mode(i, mode_vals[walker_i*num_modes + mode_i], ell, mm, &pHM[walker_i], freq_geom, &pAmp[walker_i], &amp_prefactors[walker_i], pDPreComp_all[walker_i*num_modes + mode_i], q_all[walker_i*num_modes + mode_i], amp0[walker_i], Rholm, Taulm, t0[walker_i], phi0[walker_i], cshift, walker_i, mode_i);

}
}

#ifdef __CUDACC__
	CUDA_KERNEL
	void kernel_calculate_all_modes_wrap(ModeContainer *mode_vals,
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
	        ){
	      unsigned int mm, ell;
	      double Rholm, Taulm;
	      double freq_geom;
	      //for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
	      //     walker_i < nwalkers;
	       //    walker_i += blockDim.z * gridDim.z){

	       for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
	            mode_i < num_modes;
	            mode_i += blockDim.y * gridDim.y){

	      for (int i = blockIdx.x * blockDim.x + threadIdx.x;
	           i < length;
	           i += blockDim.x * gridDim.x){

							 kernel_calculate_all_modes(mode_vals,
							       pHM,
							       freqs,
							       M_tot_sec,
							       pAmp,
							       amp_prefactors,
							       pDPreComp_all,
							       q_all,
							       amp0,
							       mode_i,
							       t0,
							       phi0,
							       cshift,
							 	     walker_i,
							 	     i,
									 	num_modes,
									length);

	}
	}
//	}
	  }
#else
void cpu_calculate_all_modes_wrap(ModeContainer *mode_vals,
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
				){
			unsigned int mm, ell;
			double Rholm, Taulm;
			double freq_geom;

			#pragma omp parallel for collapse(3)
			for (int walker_i = 0;
					 walker_i < nwalkers;
					 walker_i += 1){

			 for (int mode_i = 0;
						mode_i < num_modes;
						mode_i += 1){

			for (int i = 0;
					 i < length;
					 i += 1){

						 kernel_calculate_all_modes(mode_vals,
									 pHM,
									 freqs,
									 M_tot_sec,
									 pAmp,
									 amp_prefactors,
									 pDPreComp_all,
									 q_all,
									 amp0,
									 mode_i,
									 t0,
									 phi0,
									 cshift,
									 walker_i,
									 i,
									num_modes,
								length);

}
}
}
	}
#endif




  CUDA_CALLABLE_MEMBER
  void calculate_each_mode_PhenomD(int i, ModeContainer mode_val,
       double freq_geom,
       PhenDAmpAndPhasePreComp pDPreComp,
       double amp0, double t0, double phi0, double *cshift, double Mf_ref){
           double Rholm=1.0, Taulm=1.0;
           double phase_term1, phase_term2;
           double amp, phase;
           int status_in_for;
           UsefulPowers powers_of_f;

           int retcode = 0;

           double Mf = freq_geom;

           status_in_for = init_useful_powers(&powers_of_f, Mf);
                /*if (PD_SUCCESS != status_in_for)
                {
                  //printf("init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
                  retcode = status_in_for;
                  //exit(0);
                }
                else
                {*/
          amp = IMRPhenDAmplitude(Mf, &pDPreComp.pAmp, &powers_of_f, &pDPreComp.amp_prefactors);
               // }

               mode_val.amp[i] = amp*amp0;

              /* Add complex phase shift depending on 'm' mode */
              phase = IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm);

              Mf = freq_geom;
              phase_term1 = - t0 * (Mf - Mf_ref);
              phase_term2 = phase - (2 * phi0);

              mode_val.phase[i] = (phase_term1 + phase_term2);

  }



  /*CUDA_KERNEL
  void kernel_calculate_all_modes_PhenomD(ModeContainer *mode_vals,
        double *freqs,
        double M_tot_sec,
        PhenDAmpAndPhasePreComp *pDPreComp_all,
        double amp0,
        int num_modes,
        double t0,
        double phi0,
        double *cshift,
        int num_points
          ){
        double freq_geom;
        double Mf_ref;

        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        // if (mode_i >= num_modes) return;
         //for (int i = blockIdx.y * blockDim.x + threadIdx.x;
          //  i < length;
            //i += blockDim.x * gridDim.y)
        if (i < num_points) // kernel setup should always make second part true
        {
           freq_geom = freqs[i]*M_tot_sec;
           Mf_ref = pDPreComp_all[0].pAmp.fmaxCalc*M_tot_sec;
           calculate_each_mode_PhenomD(i, mode_vals[0], freq_geom, pDPreComp_all[0], amp0, t0, phi0, cshift, Mf_ref);

        }
    }
*/
