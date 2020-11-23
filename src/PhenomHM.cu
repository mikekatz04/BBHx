/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.


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
#include <math.h>
#include <complex>
#include <iostream>
#include "stdio.h"

#include <stdbool.h>
#include "assert.h"
#include "globalPhenomHM.h"
#include "PhenomHM.h"
#include "RingdownCW.h"
#include "IMRPhenomD_internals.h"
#include "IMRPhenomD.h"
#include "IMRPhenomD.h"
//#include "PhenomHM_internals.h"

/**
 *
 */
int PhenomHM_init_useful_mf_powers(PhenomHMUsefulMfPowers *p, double number)
{
    assert (0 != p) ; //, PD_EFAULT, "p is NULL");
    assert (!(number < 0)) ; //, PD_EDOM, "number must be non-negative");

    // consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
    p->itself = number;
    p->sqrt = sqrt(number);
    p->sixth = cbrt(p->sqrt);
    p->m_sixth = 1.0 / p->sixth;
    p->third = p->sixth * p->sixth;
    p->two_thirds = p->third * p->third;
    p->four_thirds = number * p->third;
    p->five_thirds = number * p->two_thirds;
    p->two = number * number;
    p->seven_thirds = p->third * p->two;
    p->eight_thirds = p->two_thirds * p->two;
    p->m_seven_sixths = p->m_sixth / number;
    p->m_five_sixths = p->m_seven_sixths * p->third;
    p->m_sqrt = 1. / p->sqrt;

    return 1;
}

int PhenomHM_init_useful_powers(PhenomHMUsefulPowers *p, double number)
{
    assert (0 != p) ;
    assert (number >= 0) ; //, PD_EDOM, "number must be non-negative");

    // consider changing pow(x,1/6.0) to cbrt(x) and sqrt(x) - might be faster
    double sixth = pow(number, 1.0 / 6.0);
    p->third = sixth * sixth;
    p->two_thirds = p->third * p->third;
    p->four_thirds = number * p->third;
    p->five_thirds = p->four_thirds * p->third;
    p->two = number * number;
    p->seven_thirds = p->third * p->two;
    p->eight_thirds = p->two_thirds * p->two;
    p->inv = 1. / number;
    double m_sixth = 1.0 / sixth;
    p->m_seven_sixths = p->inv * m_sixth;
    p->m_third = m_sixth * m_sixth;
    p->m_two_thirds = p->m_third * p->m_third;
    p->m_five_thirds = p->inv * p->m_two_thirds;

    return 1;
}


/**
 * returns the real and imag parts of the complex ringdown frequency
 * for the (l,m) mode.
 */
CUDA_CALLABLE_MEMBER
int IMRPhenomHMGetRingdownFrequency(
    double *fringdown,
    double *fdamp,
    unsigned int ell,
    int mm,
    double finalmass,
    double finalspin)
{
    const double inv2Pi = 0.5 / PI;
    cmplx ZZ;
    ZZ = SimRingdownCW_CW07102016(SimRingdownCW_KAPPA(finalspin, ell, mm), ell, mm, 0);
    const double Mf_RD_tmp = inv2Pi * gcmplx::real(ZZ); /* GW ringdown frequency, converted from angular frequency */
    *fringdown = Mf_RD_tmp / finalmass;         /* scale by predicted final mass */
    /* lm mode ringdown damping time (imaginary part of ringdown), geometric units */
    const double f_DAMP_tmp = inv2Pi * gcmplx::imag(ZZ); /* this is the 1./tau in the complex QNM */
    *fdamp = f_DAMP_tmp / finalmass;             /* scale by predicted final mass */
    return 1;
}

/**
 * Precompute a bunch of PhenomHM related quantities and store them filling in a
 * PhenomHMStorage variable
 */
CUDA_CALLABLE_MEMBER
static void init_PhenomHM_Storage(
    PhenomHMStorage *p,
    const double m1_SI,
    const double m2_SI,
    const double chi1z,
    const double chi2z,
    RealVector *freqs,
    const double deltaF,
    const double f_ref,
    const double phiRef,
    const double finspin,
    const double finmass
)
{

    int retcode;

    p->m1 = m1_SI / MSUN_SI;
    p->m2 = m2_SI / MSUN_SI;
    p->m1_SI = m1_SI;
    p->m2_SI = m2_SI;
    p->Mtot = p->m1 + p->m2;
    p->eta = p->m1 * p->m2 / (p->Mtot * p->Mtot);
    p->chi1z = chi1z;
    p->chi2z = chi2z;
    p->phiRef = phiRef;

    retcode = PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
        &(p->m1),
        &(p->m2),
        &(p->chi1z),
        &(p->chi2z));

    p->Mf_ref = PhenomUtilsHztoMf(p->f_ref, p->Mtot);

    p->finmass = finmass;
    p->finspin = finspin; /* dimensionless final spin */

    return 1;
};

/**
 * domain mapping function - ringdown
 */
CUDA_CALLABLE_MEMBER
double IMRPhenomHMTrd(
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
        double Rholm = pHM->Rholmlm;
        ans = Rholm * Mf; /* Used for the Phase */
    }

    return ans;
}

/**
 * mathematica function Ti
 * domain mapping function - inspiral
 */
CUDA_CALLABLE_MEMBER
double IMRPhenomHMTi(double Mf, const int mm)
{
    return 2.0 * Mf / mm;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
CUDA_CALLABLE_MEMBER
int IMRPhenomHMSlopeAmAndBm(
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
    double Trd = IMRPhenomHMTrd(fr, Mf_RD_22, Mf_RD_lm, AmpFlag, ell, mm, pHM);
    double Ti = IMRPhenomHMTi(fi, mm);

    //Am = ( Trd[fr]-Ti[fi] )/( fr - fi );
    *Am = (Trd - Ti) / (fr - fi);

    //Bm = Ti[fi] - fi*Am;
    *Bm = Ti - fi * (*Am);

    return 1;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
CUDA_CALLABLE_MEMBER
int IMRPhenomHMMapParams(
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
CUDA_CALLABLE_MEMBER
int IMRPhenomHMFreqDomainMapParams(
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
    //assert (a != NULL) ; //, PD_EFAULT, "a error");
    //assert (b != NULL) ; //, PD_EFAULT, "b error");
    //assert (fi != NULL) ; //, PD_EFAULT, "fi error");
    //assert (fr != NULL) ; //, PD_EFAULT, "fr error");
    //assert (f1 != NULL) ; //, PD_EFAULT, "f1 error");

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
    double Mf_RD_lm = pHM->Mf_RD_lm;

    // Define a ratio of QNM frequencies to be used for scaling various quantities
    double Rholm = pHM->Rholm;

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
    IMRPhenomHMSlopeAmAndBm(&Am, &Bm, mm, *fi, *fr, Mf_RD_22, Mf_RD_lm, AmpFlag, ell, pHM);

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
    int ret = IMRPhenomHMMapParams(a, b, flm, *fi, *fr, Ai, Bi, Am, Bm, Ar, Br);

    return 1;
}

/**
 * IMRPhenomHMFreqDomainMap
 * Input waveform frequency in Geometric units (Mflm)
 * and computes what frequency this corresponds
 * to scaled to the 22 mode.
 */
CUDA_CALLABLE_MEMBER
double IMRPhenomHMFreqDomainMap(
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
    int ret = IMRPhenomHMFreqDomainMapParams(&a, &b, &fi, &fr, &f1, Mflm, ell, mm, pHM, AmpFlag);

    double Mf22 = a * Mflm + b;
    return Mf22;
}

CUDA_CALLABLE_MEMBER
int IMRPhenomHMPhasePreComp(
    HMPhasePreComp *q,          /**< [out] HMPhasePreComp struct */
    const int ell,             /**< ell spherical harmonic number */
    const int mm,              /**< m spherical harmonic number */
    PhenomHMStorage *pHM,       /**< PhenomHMStorage struct */
    PhenDAmpAndPhasePreComp pDPreComp
)
{
    double ai = 0.0;
    double bi = 0.0;
    double am = 0.0;
    double bm = 0.0;
    double ar = 0.0;
    double br = 0.0;
    double fi = 0.0;
    double f1 = 0.0;
    double fr = 0.0;

    const int AmpFlag = 0;

    /* NOTE: As long as Mfshit isn't >= fr then the value of the shift is arbitrary. */
    const double Mfshift = 0.0001;

    int ret = IMRPhenomHMFreqDomainMapParams(&ai, &bi, &fi, &fr, &f1, Mfshift, ell, mm, pHM, AmpFlag);

    q->ai = ai;
    q->bi = bi;

    ret = IMRPhenomHMFreqDomainMapParams(&am, &bm, &fi, &fr, &f1, fi + Mfshift, ell, mm, pHM, AmpFlag);

    q->am = am;
    q->bm = bm;

    ret = IMRPhenomHMFreqDomainMapParams(&ar, &br, &fi, &fr, &f1, fr + Mfshift, ell, mm, pHM, AmpFlag);

    q->ar = ar;
    q->br = br;

    q->fi = fi;
    q->fr = fr;

    double Rholm = pHM->Rholm;
    double Taulm = pHM->Taulm;

    double PhDBMf = am * fi + bm;
    q->PhDBconst = IMRPhenomDPhase_OneFrequency(PhDBMf, pDPreComp, Rholm, Taulm) / am;

    double PhDCMf = ar * fr + br;
    q->PhDCconst = IMRPhenomDPhase_OneFrequency(PhDCMf, pDPreComp, Rholm, Taulm) / ar;

    double PhDBAMf = ai * fi + bi;
    q->PhDBAterm = IMRPhenomDPhase_OneFrequency(PhDBAMf, pDPreComp, Rholm, Taulm) / ai;
    return 1;
}

/**
 * Define function for FD PN amplitudes
 */
CUDA_CALLABLE_MEMBER
double IMRPhenomHMOnePointFiveSpinPN(
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
    agcmplx Hlm(0.0, 0.0);
    double M_INPUT = M1 + M2;
    M1 = M1 / (M_INPUT);
    M2 = M2 / (M_INPUT);
    double M = M1 + M2;
    double eta = M1 * M2 / (M * M);
    double delta = sqrt(1.0 - 4 * eta);
    double Xs = 0.5 * (X1z + X2z);
    double Xa = 0.5 * (X1z - X2z);
    double ans = 0;
    agcmplx I(0.0, 1.0);

    // Define PN parameter and realed powers
    double v = pow(M * 2.0 * PI * fM / m, 1.0 / 3.0);
    double v2 = v * v;
    double v3 = v * v2;

    // Define Leading Order Ampitude for each supported multipole
    if (l == 2 && m == 2)
    {
        // (l,m) = (2,2)
        // THIS IS LEADING ORDER
        Hlm = 1.0;
    }
    else if (l == 2 && m == 1)
    {
        // (l,m) = (2,1)
        // SPIN TERMS ADDED

        // UP TO 4PN
        double v4 = v * v3;
        Hlm = (sqrt(2.0) / 3.0) * \
            ( \
                v * delta - v2 * 1.5 * (Xa + delta * Xs) + \
                v3 * delta * ((335.0 / 672.0) + (eta * 117.0 / 56.0)
            ) \
            + \
            v4 * \
                ( \
                Xa * (3427.0 / 1344 - eta * 2101.0 / 336) + \
                delta * Xs * (3427.0 / 1344 - eta * 965 / 336) + \
                delta * (-I * 0.5 - PI - 2.0 * I * 0.69314718056) \
                )
            );
    }
    else if (l == 3 && m == 3)
    {
        // (l,m) = (3,3)
        // THIS IS LEADING ORDER
        Hlm = 0.75 * sqrt(5.0 / 7.0) * (v * delta);
    }
    else if (l == 3 && m == 2)
    {
        // (l,m) = (3,2)
        // NO SPIN TERMS to avoid roots
        Hlm = (1.0 / 3.0) * sqrt(5.0 / 7.0) * (v2 * (1.0 - 3.0 * eta));
    }
    else if (l == 4 && m == 4)
    {
        // (l,m) = (4,4)
        // THIS IS LEADING ORDER
        Hlm = (4.0 / 9.0) * sqrt(10.0 / 7.0) * v2 * (1.0 - 3.0 * eta);
    }
    else if (l == 4 && m == 3)
    {
        // (l,m) = (4,3)
        // NO SPIN TERMS TO ADD AT DESIRED ORDER
        Hlm = 0.75 * sqrt(3.0 / 35.0) * v3 * delta * (1.0 - 2.0 * eta);
    }
    else
    {
        printf("requested ell = %i and m = %i mode not available, check documentation for available modes\n", l, m);
        assert(0); //ERROR(PD_EDOM, "error");
    }
    // Compute the final PN Amplitude at Leading Order in fM
    ans = M * M * PI * sqrt(eta * 2.0 / 3) * pow(v, -3.5) * gcmplx::abs(Hlm);

    return ans;
}


/** @} */
/** @} */


 /**
  * Michael Katz added this function.
  * internal function that filles amplitude and phase for a specific frequency and mode.
  */
 CUDA_CALLABLE_MEMBER
 void calculate_mode(int binNum, int mode_i, double* amps, double* phases, int ell, int mm, PhenomHMStorage *pHM, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double amp0, double Rholm, double Taulm, double t0, double phi0, int length, int numBinAll, int numModes)
 {
         double freq_amp, Mf, beta_term1, beta, beta_term2, HMamp_term1, HMamp_term2;
         double Mf_wf, Mfr, tmpphaseC, phase_term1, phase_term2;
         double amp_i, phase_i;
         int status_in_for;
         UsefulPowers powers_of_f;
         int retcode = 0;

         for (int i = 0; i < length; i += 1)
         {
             int mode_index = (i * mode_i + numModes) * numBinAll + binNum;
             int freq_index = i * numBinAll + binNum;
             freq_geom = freqs[freq_index]*M_tot_sec;

             // generate amplitude
             // IMRPhenomHMAmplitude
             freq_amp = IMRPhenomHMFreqDomainMap(freq_geom, ell, mm, pHM, AmpFlagTrue);
             Mf = freq_amp; // geometric frequency

             status_in_for = init_useful_powers(&powers_of_f, Mf);
             amp_i = IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, &amp_prefactors);

             int retcode = 0;

              /* loop over only positive m is intentional. negative m added automatically */
              // generate amplitude
              // IMRPhenomHMAmplitude

              freq_amp = IMRPhenomHMFreqDomainMap(freq_geom, ell, mm, pHM, AmpFlagTrue);
              Mf = freq_amp; // geometric frequency

              status_in_for = init_useful_powers(&powers_of_f, Mf);

              amp_i = IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, amp_prefactors);

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
              if (beta_term1 == 0.)
              {
                  beta = 0.;
              }
              else
              {
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

              amps[mode_index] = amp_i*amp0;

              Mf_wf = 0.0;
              Mf = 0.0;
              Mfr = 0.0;
              tmpphaseC = 0.0;

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

              Mf = freq_geom;
              phase_term1 = - t0 * (Mf - pHM->Mf_ref);
              phase_term2 = phase_i - (mm * phi0);

              phases[mode_index] = (phase_term1 + phase_term2);
         }
}




/**
 * Michael Katz added this function.
 * Main function for calculating PhenomHM in the form used by Michael Katz
 * This is setup to allow for pre-allocation of arrays. Therefore, all arrays
 * should be setup outside of this function.
 */

CUDA_CALLABLE_MEMBER
int IMRPhenomHMCore(
    int *ells,
    int *mms,
    double* amps,
    double* phases,
    double* freqs,                      /**< GW frequecny list [Hz] */
    double m1_SI,                               /**< primary mass [kg] */
    double m2_SI,                               /**< secondary mass [kg] */
    double chi1z,                               /**< aligned spin of primary */
    double chi2z,                               /**< aligned spin of secondary */
    const double distance,                      /**< distance [m] */
    const double phiRef,                        /**< orbital phase at f_ref */
    double f_ref,
    double finspin,
    double finmass,
    int length,                              /**< reference GW frequency */
    int num_modes,
    int binNum,
    int numBinAll,
)
{
    double t0, amp0, phi0;
    /* setup PhenomHM model storage struct / structs */
    /* Compute quantities/parameters related to PhenomD only once and store them */
    //PhenomHMStorage *pHM;
    PhenomHMStorage pHM;
    init_PhenomHM_Storage(
        &pHM,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        freqs,
        deltaF,
        f_ref,
        phiRef,
        finspin,
        finmass
    );


    /* populate the ringdown frequency array */
    /* If you want to model a new mode then you have to add it here. */
    /* (l,m) = (2,2) */

    double fring22, fdamp22;

    IMRPhenomHMGetRingdownFrequency(
        &pHM->Mf_RD_22,
        &ppHM->Mf_DM_22,
        2, 2,
        pHM->finmass, pHM->finspin);



    /* (l,m) = (2,2) */
    int ell, mm;
    ell = 2;
    mm = 2;
    pHM->Rholm22 = 1.0;
    pHM->Taulm22 = 1.0;


    // Prepare 22 coefficients
    PhenDAmpAndPhasePreComp pDPreComp22;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp22,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm22,
        pHM->Taulm22);

    // set f_ref to f_max
    if (pHM->f_ref == 0.0){
        pHM->Mf_ref = pDPreComp22.pAmp.fmaxCalc;
        pHM->f_ref = PhenomUtilsMftoHz(pHM->Mf_ref, pHM->Mtot);
        //printf("%e, %e\n", pHM->f_ref, pHM->Mf_ref);
    }

    /* compute the reference phase shift need to align the waveform so that
     the phase is equal to phiRef at the reference frequency f_ref. */
    /* the phase shift is computed by evaluating the phase of the
    (l,m)=(2,2) mode.
    phi0 is the correction we need to add to each mode. */
    double phiRef_to_zero = 0.0;
    double phi_22_at_f_ref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,  1.0, 1.0);

    // phi0 is passed into this function as a pointer.This is for compatibility with GPU.
    phi0[0] = 0.5 * (phi_22_at_f_ref + phiRef_to_zero); // TODO: check this, I think it should be half of phiRef as well

    // t0 is passed into this function as a pointer.This is for compatibility with GPU.
    t0[binNum] = IMRPhenomDComputet0(pHM->eta, pHM->chi1z, pHM->chi2z, pHM->finspin, pDPreComp22->pPhi, pDPreComp22->pAmp);

    // setup PhenomD info. Sub here is due to preallocated struct
    pAmp_trans = pDPreComp22->pAmp;

    retcode = 0;
    AmpInsPrefactors amp_prefactors = pDPreComp22->amp_prefactors;

    const double Mtot = (m1_SI + m2_SI) / MSUN_SI;

   /* Compute the amplitude pre-factor */
   // amp0 is passed into this function as a pointer.This is for compatibility with GPU.
   amp0[binNum] = PhenomUtilsFDamp0(Mtot, distance); // TODO check if this is right units

    //HMPhasePreComp q;

    // prep q and pDPreComp for each mode in the loop below
    HMPhasePreComp qlm;
    PhenDAmpAndPhasePreComp pDPreComplm;

    double Rholm, Taulm;
    unsigned int ell, mm;

    for (int mode_i=0; mode_i<num_modes; mode_i++){
        ell = ells[mode_i];
        mm = mms[mode_i];

        IMRPhenomHMGetRingdownFrequency(
            &pHM->Mf_RD_mode,
            &pHM->Mf_DM_mode,
            ell, mm,
            pHM->finmass, pHM->finspin);

        pHM->Rholm = pHM->Mf_RD_22 / pHM->Mf_RD_mode;
        pHM->Taulm = pHM->Mf_DM_mode / pHM->Mf_DM_22;

        retcode = 0;

        Rholm = pHM->Rholm;
        Taulm = pHM->Taulm;
        retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
            &pDPreComplm,
            pHM->m1,
            pHM->m2,
            pHM->chi1z,
            pHM->chi2z,
            Rholm,
            Taulm);

        retcode = IMRPhenomHMPhasePreComp(&qlm, ell, mm, &pHM, pDPreComplm);

    calculate_modes(mode_vals, pHM, freqs_trans, M_tot_sec, pAmp, *amp_prefactors, pDPreComp_all, q_all, amp0[0], num_modes, t0[0], phi0[0]);

    }
}


/**
 * @addtogroup LALSimIMRPhenom_c
 * @{
 *
 * @name Routines for IMR Phenomenological Model "HM"
 * @{
 *
 * @author Sebastian Khan, Francesco Pannarale, Lionel London
 *
 * @brief C code for IMRPhenomHM phenomenological waveform model.
 *
 * Inspiral-merger and ringdown phenomenological, frequecny domain
 * waveform model for binary black holes systems.
 * Models not only the dominant (l,|m|) = (2,2) modes
 * but also some of the sub-domant modes too.
 * Model described in PhysRevLett.120.161102/1708.00404.
 * The model is based on IMRPhenomD (\cite Husa:2015iqa, \cite Khan:2015jqa)
 *
 * @note The higher mode information was not calibrated to Numerical Relativity
 * simulation therefore the calibration range is inherited from PhenomD.
 *
 * @attention The model is usable outside this parameter range,
 * and in tests to date gives sensible physical results,
 * but conclusive statements on the physical fidelity of
 * the model for these parameters await comparisons against further
 * numerical-relativity simulations. For more information, see the review wiki
 * under https://git.ligo.org/waveforms/reviews/phenomhm/wikis/home
 * Also a technical document in the DCC https://dcc.ligo.org/LIGO-T1800295
 */

/**
 * Original Code: Returns h+ and hx in the frequency domain.
 *
 * This function can be called in the usual sense
 * where you supply a f_min, f_max and deltaF.
 * This is the case when deltaF > 0.
 * If f_max = 0. then the default ending frequnecy is used.
 * or you can also supply a custom set of discrete
 * frequency points with which to evaluate the waveform.
 * To do this you must call this function with
 * deltaF <= 0.
 *
 */

 #define MAX_MODES 6

 CUDA_KERNEL
 void IMRPhenomHM(
     double* t0,
     double* amp0,
     double* phi0,
     double* mode_vals, /**< [out] Frequency-domain waveform hx */
     int* ells_in,
     int* mms_in
     double* freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
     double* m1_SI,                        /**< mass of companion 1 (kg) */
     double* m2_SI,                        /**< mass of companion 2 (kg) */
     double* chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
     double* chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
     double* distance,               /**< distance of source (m) */
     double* phiRef,                 /**< reference orbital phase (rad) */
     double* f_ref,                        /**< Reference frequency */
     int num_modes,
     int length,
     int numBinAll,
)
{

    /*
     * Phase shift due to leading order complex amplitude
     * [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
     * "Spherical hrmonic modes for numerical relativity"
     */
    /* List of phase shifts: the index is the azimuthal number m */
    __shared__ double cShift[7];

    __shared__ int ells[MAX_MODES];
    __shared__ int ms[MAX_MODES];

    if (threadIdx.x == 0)
    {
        cShift[0] = 0.0;
        cShift[1] = PI_2; /* i shift */
        cShift[2] = 0.0;
        cShift[3] = -PI_2; /* -i shift */
        cShift[4] = PI; /* 1 shift */
        cShift[5] = PI_2; /* -1 shift */
        cShift[6] = 0.0;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_modes; i += blockDim.x)
    {
        ells[i] = ells_in[i];
        mms[i] = mms_in[i];
    }

    __syncthreads();

    int binNum = threadIdx.x + blockDim.x * blockIdx.x;

    if (binNum < numBinAll)
    {
        IMRPhenomHMCore(ells, mms, amps, phases, freqs, m1_SI[binNum], m2_SI[binNum], chi1z[binNUM], chi2z[binNUM], distance[binNUM], phiRef[binNUM], f_ref[binNUM], finspin[binNUM], finmass[binNUM], length, numModes, binNum, numBinAll);
    }
}

int main(){

    int f_length = 1024;
    double *freqs = (double*)malloc(f_length*sizeof(double));
    double m1_SI = 3e7*1.989e30;
    double m2_SI = 1e7*1.989e30;
    double chi1z = 0.8;
    double chi2z = 0.8;
    double distance = 1.0e9*3.086e16;
    double phiRef = 0.0;
    double f_ref = 1e-4;
    double deltaF = -1.0;
    int num_modes = 4;
    unsigned int l[num_modes];
    unsigned int m[num_modes];

    m[0] = 2;
    l[0] = 2;
    m[1] = 1;
    l[1] = 2;
    m[2] = 3;
    l[2] = 3;
    m[3] = 4;
    l[3] = 4;

    int num_walkers = 2;
    int to_gpu = 0;
    int to_interp = 0;
    ModeContainer * mode_vals = cpu_create_modes(num_modes, num_walkers, l, m, f_length, to_gpu, to_interp);
    int i;
    for (i=0; i<num_modes; i++){
        mode_vals[i].length = f_length;
    }

    for (i=0; i<f_length; i++){
        freqs[i] = 1e-4*(i+1);
        if (i % 100 == 0) printf("%e\n", freqs[i]);
    }

int out = IMRPhenomHM(
    mode_vals, /**< [out] Frequency-domain waveform hx */
    freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
    f_length,
    m1_SI,                        /**< mass of companion 1 (kg) */
    m2_SI,                        /**< mass of companion 2 (kg) */
    chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
    chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
    distance,               /**< distance of source (m) */
    phiRef,                 /**< reference orbital phase (rad) */
    deltaF,                 /**< Sampling frequency (Hz). To use arbitrary frequency points set deltaF <= 0. */
    f_ref,                        /**< Reference frequency */
    num_modes,
    to_gpu);

int j;
for (i=0; i<num_modes; i++){
    for (j=0; j<f_length; j++){
        if (j % 100 == 0) printf("%e, %e, %e\n", freqs[j], mode_vals[i].amp[j], mode_vals[i].phase[j]);
    }
}
//printf("%e\n", pHM_trans->m1);

free(mode_vals);
free(freqs);
//free(pHM_trans);
return(0);
}
