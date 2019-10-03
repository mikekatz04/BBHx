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
#include <cuComplex.h>

#include "globalPhenomHM.h"


__device__
int d_init_useful_powers(UsefulPowers *p, double number)
{
	//CHECK(0 != p, PD_EFAULT, "p is NULL");
	//CHECK(number >= 0, PD_EDOM, "number must be non-negative");

	// consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
	p->sixth = pow(number, 1/6.0);
	p->third = p->sixth * p->sixth;
	p->two_thirds = number / p->third;
	p->four_thirds = number * (p->third);
	p->five_thirds = p->four_thirds * (p->third);
	p->two = number * number;
	p->seven_thirds = p->third * p->two;
	p->eight_thirds = p->two_thirds * p->two;

	return 1;
}

/**
 * domain mapping function - ringdown
 */
 __device__
double d_IMRPhenomHMTrd(
    double Mf,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM)
{
    double ans = 0.0;
    if (AmpFlag == AmpFlagTrue)
    {
        /* For amplitude */
        ans = Mf - Mf_RD_lm + Mf_RD_22; /*Used for the Amplitude as an approx fix for post merger powerlaw slope */
    }
    else
    {
        /* For phase */
        double Rholm = pHM->Rholm[ell][mm];
        ans = Rholm * Mf; /* Used for the Phase */
    }

    return ans;
}

/**
 * mathematica function Ti
 * domain mapping function - inspiral
 */
 __device__
double d_IMRPhenomHMTi(double Mf, const int mm)
{
    return 2.0 * Mf / mm;
}


/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
__device__
int d_IMRPhenomHMSlopeAmAndBm(
    double *Am,
    double *Bm,
    const int mm,
    double fi,
    double fr,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    PhenomHMStorage *pHM)
{
    double Trd = d_IMRPhenomHMTrd(fr, Mf_RD_22, Mf_RD_lm, AmpFlag, ell, mm, pHM);
    double Ti = d_IMRPhenomHMTi(fi, mm);

    //Am = ( Trd[fr]-Ti[fi] )/( fr - fi );
    *Am = (Trd - Ti) / (fr - fi);

    //Bm = Ti[fi] - fi*Am;
    *Bm = Ti - fi * (*Am);

    return 1;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
__device__
int d_IMRPhenomHMMapParams(
    double *a,
    double *b,
    double flm,
    double fi,
    double fr,
    double Ai,
    double Bi,
    double Am,
    double Bm,
    double Ar,
    double Br)
{
    // Define function to output map params used depending on
    if (flm > fi)
    {
        if (flm > fr)
        {
            *a = Ar;
            *b = Br;
        }
        else
        {
            *a = Am;
            *b = Bm;
        }
    }
    else
    {
        *a = Ai;
        *b = Bi;
    };
    return 1;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
__device__
int d_IMRPhenomHMFreqDomainMapParams(
    double *a,             /**< [Out]  */
    double *b,             /**< [Out]  */
    double *fi,            /**< [Out]  */
    double *fr,            /**< [Out]  */
    double *f1,            /**< [Out]  */
    const double flm,      /**< input waveform frequency */
    const int ell,       /**< spherical harmonics ell mode */
    const int mm,        /**< spherical harmonics m mode */
    PhenomHMStorage *pHM, /**< Stores quantities in order to calculate them only once */
    const int AmpFlag     /**< is ==1 then computes for amplitude, if ==0 then computes for phase */
)
{

    /*check output points are NULL*/
    //CHECK(a != NULL, PD_EFAULT, "a error");
    //CHECK(b != NULL, PD_EFAULT, "b error");
    //CHECK(fi != NULL, PD_EFAULT, "fi error");
    //CHECK(fr != NULL, PD_EFAULT, "fr error");
    //CHECK(f1 != NULL, PD_EFAULT, "f1 error");

    /* Account for different f1 definition between PhenomD Amplitude and Phase derivative models */
    double Mf_1_22 = 0.; /* initalise variable */
    if (AmpFlag == AmpFlagTrue)
    {
        /* For amplitude */
        Mf_1_22 = AMP_fJoin_INS; /* inspiral joining frequency from PhenomD [amplitude model], for the 22 mode */
    }
    else
    {
        /* For phase */
        Mf_1_22 = PHI_fJoin_INS; /* inspiral joining frequency from PhenomD [phase model], for the 22 mode */
    }

    *f1 = Mf_1_22;

    double Mf_RD_22 = pHM->Mf_RD_22;
    double Mf_RD_lm = pHM->PhenomHMfring[ell][mm];

    // Define a ratio of QNM frequencies to be used for scaling various quantities
    double Rholm = pHM->Rholm[ell][mm];

    // Given experiments with the l!=m modes, it appears that the QNM scaling rather than the PN scaling may be optimal for mapping f1
    double Mf_1_lm = Mf_1_22 / Rholm;

    /* Define transition frequencies */
    *fi = Mf_1_lm;
    *fr = Mf_RD_lm;

    /*Define the slope and intercepts of the linear transformation used*/
    double Ai = 2.0 / mm;
    double Bi = 0.0;
    double Am;
    double Bm;
    d_IMRPhenomHMSlopeAmAndBm(&Am, &Bm, mm, *fi, *fr, Mf_RD_22, Mf_RD_lm, AmpFlag, ell, pHM);

    double Ar = 1.0;
    double Br = 0.0;
    if (AmpFlag == AmpFlagTrue)
    {
        /* For amplitude */
        Br = -Mf_RD_lm + Mf_RD_22;
    }
    else
    {
        /* For phase */
        Ar = Rholm;
    }

    /* Define function to output map params used depending on */
    int ret = d_IMRPhenomHMMapParams(a, b, flm, *fi, *fr, Ai, Bi, Am, Bm, Ar, Br);
    if (ret != 1)
    {
        //printf("IMRPhenomHMMapParams failed in IMRPhenomHMFreqDomainMapParams (1)\n");
        //ERROR(PD_EDOM, "error");
    }

    return 1;
}

/**
 * IMRPhenomHMFreqDomainMap
 * Input waveform frequency in Geometric units (Mflm)
 * and computes what frequency this corresponds
 * to scaled to the 22 mode.
 */
__device__
double d_IMRPhenomHMFreqDomainMap(
    double Mflm,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM,
    const int AmpFlag)
{

    /* Mflm here has the same meaning as Mf_wf in IMRPhenomHMFreqDomainMapHM (old deleted function). */
    double a = 0.;
    double b = 0.;
    /* Following variables not used in this funciton but are returned in IMRPhenomHMFreqDomainMapParams */
    double fi = 0.;
    double fr = 0.;
    double f1 = 0.;
    int ret = d_IMRPhenomHMFreqDomainMapParams(&a, &b, &fi, &fr, &f1, Mflm, ell, mm, pHM, AmpFlag);
    if (ret != 1)
    {
        //printf("IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMap\n");
        //ERROR(PD_EDOM, "error");
    }
    double Mf22 = a * Mflm + b;
    return Mf22;
}


__device__ __forceinline__ cuDoubleComplex my_cexpf (cuDoubleComplex z)

{



    cuDoubleComplex t = make_cuDoubleComplex (exp (cuCreal(z)), 0.0);

    cuDoubleComplex v = make_cuDoubleComplex (cos(cuCimag(z)), sin(cuCimag(z)));
    cuDoubleComplex res = cuCmul(t, v);
    return res;

}


__device__ double complex_norm(double real, double imag){
   return sqrt(real*real + imag*imag);
}

__device__
double d_IMRPhenomHMOnePointFiveSpinPN(
    double fM,
    int l,
    int m,
    double M1,
    double M2,
    double X1z,
    double X2z)
{

    // LLondon 2017

    // Define effective intinsic parameters
    double Hlm_real = 0.0;
    double Hlm_imag = 0.0;
    double M_INPUT = M1 + M2;
    M1 = M1 / (M_INPUT);
    M2 = M2 / (M_INPUT);
    double M = M1 + M2;
    double eta = M1 * M2 / (M * M);
    double delta = sqrt(1.0 - 4 * eta);
    double Xs = 0.5 * (X1z + X2z);
    double Xa = 0.5 * (X1z - X2z);
    double ans = 0;

    // Define PN parameter and realed powers
    double v = pow(M * 2.0 * PI * fM / m, 1.0 / 3.0);
    double v2 = v * v;
    double v3 = v * v2;

    // Define Leading Order Ampitude for each supported multipole
    if (l == 2 && m == 2)
    {
        // (l,m) = (2,2)
        // THIS IS LEADING ORDER
        Hlm_real = 1.0;
    }
    else if (l == 2 && m == 1)
    {
        // (l,m) = (2,1)
        // SPIN TERMS ADDED

        // UP TO 4PN
        double v4 = v * v3;
        Hlm_real = (sqrt(2.0) / 3.0) * \
            ( \
                v * delta - v2 * 1.5 * (Xa + delta * Xs) + \
                v3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)
            ) \
            + \
            v4 * \
                ( \
                Xa * (3427.0 / 1344 - eta * 2101.0 / 336) + \
                delta * Xs * (3427.0 / 1344 - eta * 965 / 336) + \
                delta * (- PI))
            );
        Hlm_imag =  (sqrt(2.0) / 3.0) * v4 *(delta * (-0.5 - 2 * 0.69314718056)); //I is in the end of this statement in each term in parentheses

    }
    else if (l == 3 && m == 3)
    {
        // (l,m) = (3,3)
        // THIS IS LEADING ORDER
        Hlm_real = 0.75 * sqrt(5.0 / 7.0) * (v * delta);
    }
    else if (l == 3 && m == 2)
    {
        // (l,m) = (3,2)
        // NO SPIN TERMS to avoid roots
        Hlm_real = (1.0 / 3.0) * sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta));
    }
    else if (l == 4 && m == 4)
    {
        // (l,m) = (4,4)
        // THIS IS LEADING ORDER
        Hlm_real = (4.0 / 9.0) * sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta);
    }
    else if (l == 4 && m == 3)
    {
        // (l,m) = (4,3)
        // NO SPIN TERMS TO ADD AT DESIRED ORDER
        Hlm_real = 0.75 * sqrt(3.0 / 35.0) * v3 * delta * (1.0 - 2.0 * eta);
    }
    else
    {
        //printf("requested ell = %i and m = %i mode not available, check documentation for available modes\n", l, m);
        //ERROR(PD_EDOM, "error");
    }
    // Compute the final PN Amplitude at Leading Order in fM
    ans = M * M * PI * sqrt(eta * 2.0 / 3) * pow(v, -3.5) * complex_norm(Hlm_real, Hlm_imag);

    return ans;
}

  __device__
  double d_AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors) {
    double Mf2 = powers_of_Mf->two;
    double Mf3 = Mf*Mf2;

    return 1 + powers_of_Mf->two_thirds * prefactors->two_thirds
  			+ Mf * prefactors->one + powers_of_Mf->four_thirds * prefactors->four_thirds
  			+ powers_of_Mf->five_thirds * prefactors->five_thirds + Mf2 * prefactors->two
  			+ powers_of_Mf->seven_thirds * prefactors->seven_thirds + powers_of_Mf->eight_thirds * prefactors->eight_thirds
  			+ Mf3 * prefactors->three;
  }

  __device__
  double d_AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
    double fRD = p->fRD;
    double fDM = p->fDM;
    double gamma1 = p->gamma1;
    double gamma2 = p->gamma2;
    double gamma3 = p->gamma3;
    double fDMgamma3 = fDM*gamma3;
    double fminfRD = f - fRD;
    return exp( -(fminfRD)*gamma2 / (fDMgamma3) )
      * (fDMgamma3*gamma1) / (pow(fminfRD, 2.0) + pow(fDMgamma3, 2.0));
  }

  __device__
  double d_AmpIntAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
    double Mf2 = Mf*Mf;
    double Mf3 = Mf*Mf2;
    double Mf4 = Mf*Mf3;
    return p->delta0 + p->delta1*Mf + p->delta2*Mf2 + p->delta3*Mf3 + p->delta4*Mf4;
  }


  // Call ComputeIMRPhenomDAmplitudeCoefficients() first!
  /**
   * This function computes the IMR amplitude given phenom coefficients.
   * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
   */
  __device__ double d_IMRPhenDAmplitude(double f, IMRPhenomDAmplitudeCoefficients *p, UsefulPowers *powers_of_f, AmpInsPrefactors * prefactors) {
    // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
    // The inspiral, intermediate and merger-ringdown amplitude parts

    // Transition frequencies
    p->fInsJoin = AMP_fJoin_INS;
    p->fMRDJoin = p->fmaxCalc;

    double f_seven_sixths = f * powers_of_f->sixth;
    double AmpPreFac = prefactors->amp0 / f_seven_sixths;

    // split the calculation to just 1 of 3 possible mutually exclusive ranges

    if (f <= p->fInsJoin)	// Inspiral range
    {
  	  double AmpIns = AmpPreFac * d_AmpInsAnsatz(f, powers_of_f, prefactors);
  	  return AmpIns;
    }

    if (f >= p->fMRDJoin)	// MRD range
    {
  	  double AmpMRD = AmpPreFac * d_AmpMRDAnsatz(f, p);
  	  return AmpMRD;
    }

    //	Intermediate range
    double AmpInt = AmpPreFac * d_AmpIntAnsatz(f, p);
    return AmpInt;
  }

  __device__
  double d_PhiInsAnsatzInt(double Mf, UsefulPowers *powers_of_Mf, PhiInsPrefactors *prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn)
  {
  	//CHECK(0 != pn, PD_EFAULT, "pn is NULL");

    // Assemble PN phasing series
    const double v = powers_of_Mf->third * pow(PI, 1./3.);
    const double logv = log(v);

    double phasing = prefactors->initial_phasing;

    phasing += prefactors->two_thirds	* powers_of_Mf->two_thirds;
    phasing += prefactors->third * powers_of_Mf->third;
    phasing += prefactors->third_with_logv * logv * powers_of_Mf->third;
    phasing += prefactors->logv * logv;
    phasing += prefactors->minus_third / powers_of_Mf->third;
    phasing += prefactors->minus_two_thirds / powers_of_Mf->two_thirds;
    phasing += prefactors->minus_one / Mf;
    phasing += prefactors->minus_five_thirds / powers_of_Mf->five_thirds; // * v^0

    // Now add higher order terms that were calibrated for PhenomD
    phasing += ( prefactors->one * Mf + prefactors->four_thirds * powers_of_Mf->four_thirds
  			   + prefactors->five_thirds * powers_of_Mf->five_thirds
  			   + prefactors->two * powers_of_Mf->two
  			 ) / p->eta;

    return phasing;
  }


  __device__
  double d_PhiMRDAnsatzInt(double f, IMRPhenomDPhaseCoefficients *p, double Rholm, double Taulm)
  {
    double sqrootf = sqrt(f);
    double fpow1_5 = f * sqrootf;
    // check if this is any faster: 2 sqrts instead of one pow(x,0.75)
    double fpow0_75 = sqrt(fpow1_5); // pow(f,0.75);

    return -(p->alpha2/f)
  		 + (4.0/3.0) * (p->alpha3 * fpow0_75)
  		 + p->alpha1 * f
  		 + p->alpha4 * Rholm * atan((f - p->alpha5 * p->fRD) / (Rholm * p->fDM * Taulm));
  }

  __device__
  double d_PhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
    // 1./eta in paper omitted and put in when need in the functions:
    // ComputeIMRPhenDPhaseConnectionCoefficients
    // IMRPhenDPhase
    return  p->beta1*Mf - p->beta3/(3.*pow(Mf, 3.0)) + p->beta2*log(Mf);
  }



__device__
double d_IMRPhenDPhase(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, UsefulPowers *powers_of_f, PhiInsPrefactors *prefactors, double Rholm, double Taulm)
{
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermendiate and merger-ringdown phase parts

  // split the calculation to just 1 of 3 possible mutually exclusive ranges
  if (f <= p->fInsJoin)	// Inspiral range
  {
	  double PhiIns = d_PhiInsAnsatzInt(f, powers_of_f, prefactors, p, pn);
	  return PhiIns;
  }

  if (f >= p->fMRDJoin)	// MRD range
  {
	  double PhiMRD = p->etaInv * d_PhiMRDAnsatzInt(f, p, Rholm, Taulm) + p->C1MRD + p->C2MRD * f;
	  return PhiMRD;
  }

  //	Intermediate range
  double PhiInt = p->etaInv * d_PhiIntAnsatz(f, p) + p->C1Int + p->C2Int * f;
  return PhiInt;
}

  __device__
   double d_IMRPhenomDPhase_OneFrequency(
      double Mf,
      PhenDAmpAndPhasePreComp pD,
      double Rholm,
      double Taulm)
  {

    UsefulPowers powers_of_f;
    int status = d_init_useful_powers(&powers_of_f, Mf);
    //CHECK(PD_SUCCESS == status, status, "Failed to initiate init_useful_powers");
    double phase = d_IMRPhenDPhase(Mf, &(pD.pPhi), &(pD.pn), &powers_of_f,
                                &(pD.phi_prefactors), Rholm, Taulm);
    return phase;
  }


__device__
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


        freq_amp = d_IMRPhenomHMFreqDomainMap(freq_geom, ell, mm, pHM, AmpFlagTrue);

            //status_in_for = PD_SUCCESS;
          /* Now generate the waveform */
              Mf = freq_amp; //freqs->data[i]; // geometric frequency

              status_in_for = d_init_useful_powers(&powers_of_f, Mf);
              /*if (PD_SUCCESS != status_in_for)
              {
                //printf("init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
                retcode = status_in_for;
                //exit(0);
              }
              else
              {*/
                amp_i = d_IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, amp_prefactors);
             // }


            beta_term1 = d_IMRPhenomHMOnePointFiveSpinPN(
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

                beta_term2 = d_IMRPhenomHMOnePointFiveSpinPN(2.0 * freq_geom / mm, ell, mm, pHM->m1, pHM->m2, pHM->chi1z, pHM->chi2z);
                beta = beta_term1 / beta_term2;

                /* LL: Apply steps #1 and #2 */
                HMamp_term1 = d_IMRPhenomHMOnePointFiveSpinPN(
                    freq_amp,
                    ell,
                    mm,
                    pHM->m1,
                    pHM->m2,
                    pHM->chi1z,
                    pHM->chi2z);
                HMamp_term2 = d_IMRPhenomHMOnePointFiveSpinPN(freq_amp, 2, 2, pHM->m1, pHM->m2, 0.0, 0.0);

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
                phase_i += d_IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.ai;
            }
            else if (!(Mf_wf > q.fr))
            { /* in mathematica -> IMRPhenDPhaseB */
                Mf = q.am * Mf_wf + q.bm;
                phase_i += d_IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.am - q.PhDBconst + q.PhDBAterm;
            }
            else if ((Mf_wf > q.fr))
            { /* in mathematica -> IMRPhenDPhaseC */
                Mfr = q.am * q.fr + q.bm;
                tmpphaseC = d_IMRPhenomDPhase_OneFrequency(Mfr, pDPreComp, Rholm, Taulm) / q.am - q.PhDBconst + q.PhDBAterm;
                Mf = q.ar * Mf_wf + q.br;
                phase_i += d_IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.ar - q.PhDCconst + tmpphaseC;
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


__global__
void kernel_calculate_all_modes(ModeContainer *mode_vals,
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
      unsigned int mode_i = blockIdx.y;
	  unsigned int walker_i = blockIdx.z;

      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      /* if (mode_i >= num_modes) return;
       for (int i = blockIdx.y * blockDim.x + threadIdx.x;
          i < length;
          i += blockDim.x * gridDim.y)*/

      if ((i < (&pHM[walker_i])->ind_max) && (i >= (&pHM[walker_i])->ind_min) && (mode_i < num_modes) && (walker_i < nwalkers))  // kernel setup should always make second part true
      {

         ell = mode_vals[walker_i*num_modes + mode_i].l;
         mm = mode_vals[walker_i*num_modes + mode_i].m;
         Rholm = (&pHM[walker_i])->Rholm[ell][mm];
         Taulm = (&pHM[walker_i])->Taulm[ell][mm];
         freq_geom = freqs[walker_i*length + i]*M_tot_sec[walker_i];

         calculate_each_mode(i, mode_vals[walker_i*num_modes + mode_i], ell, mm, &pHM[walker_i], freq_geom, &pAmp[walker_i], &amp_prefactors[walker_i], pDPreComp_all[walker_i*num_modes + mode_i], q_all[walker_i*num_modes + mode_i], amp0[walker_i], Rholm, Taulm, t0[walker_i], phi0[walker_i], cshift, walker_i, mode_i);

      }
  }




  __device__
  void calculate_each_mode_PhenomD(int i, ModeContainer mode_val,
       double freq_geom,
       PhenDAmpAndPhasePreComp pDPreComp,
       double amp0, double t0, double phi0, double *cshift, double Mf_ref){
           double Rholm=1.0, Taulm=1.0;
           double phase_term1, phase_term2;
           double amp, phase;
           int status_in_for;
           UsefulPowers powers_of_f;
           //cuDoubleComplex J = make_cuDoubleComplex(0.0, 1.0);
           int retcode = 0;

           double Mf = freq_geom;

           status_in_for = d_init_useful_powers(&powers_of_f, Mf);
                /*if (PD_SUCCESS != status_in_for)
                {
                  //printf("init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
                  retcode = status_in_for;
                  //exit(0);
                }
                else
                {*/
          amp = d_IMRPhenDAmplitude(Mf, &pDPreComp.pAmp, &powers_of_f, &pDPreComp.amp_prefactors);
               // }

               mode_val.amp[i] = amp*amp0;

              /* Add complex phase shift depending on 'm' mode */
              phase = d_IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm);

              Mf = freq_geom;
              phase_term1 = - t0 * (Mf - Mf_ref);
              phase_term2 = phase - (2 * phi0);

              mode_val.phase[i] = (phase_term1 + phase_term2);

  }



  __global__
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
        /* if (mode_i >= num_modes) return;
         for (int i = blockIdx.y * blockDim.x + threadIdx.x;
            i < length;
            i += blockDim.x * gridDim.y)*/
        if (i < num_points) // kernel setup should always make second part true
        {
           freq_geom = freqs[i]*M_tot_sec;
           Mf_ref = pDPreComp_all[0].pAmp.fmaxCalc*M_tot_sec;
           calculate_each_mode_PhenomD(i, mode_vals[0], freq_geom, pDPreComp_all[0], amp0, t0, phi0, cshift, Mf_ref);

        }
    }
