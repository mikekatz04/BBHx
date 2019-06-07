/*
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

#include <stdbool.h>
#include "assert.h"
#include "globalPhenomHM.h"
#include "PhenomHM.h"
#include "RingdownCW.h"
#include "IMRPhenomD_internals.h"
#include "IMRPhenomD.h"
#include "IMRPhenomD.h"
//#include "PhenomHM_internals.h"
/*
 * Phase shift due to leading order complex amplitude
 * [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
 * "Spherical hrmonic modes for numerical relativity"
 */
/* List of phase shifts: the index is the azimuthal number m */
static const double cShift[7] = {0.0,
                                 PI_2 /* i shift */,
                                 0.0,
                                 -PI_2 /* -i shift */,
                                 PI /* 1 shift */,
                                 PI_2 /* -1 shift */,
                                 0.0};

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
    const double Mf_RD_tmp = inv2Pi * std::real(ZZ); /* GW ringdown frequency, converted from angular frequency */
    *fringdown = Mf_RD_tmp / finalmass;         /* scale by predicted final mass */
    /* lm mode ringdown damping time (imaginary part of ringdown), geometric units */
    const double f_DAMP_tmp = inv2Pi * std::imag(ZZ); /* this is the 1./tau in the complex QNM */
    *fdamp = f_DAMP_tmp / finalmass;             /* scale by predicted final mass */
    return 1;
}

/**
 * helper function to easily check if the
 * input frequency sequence is uniformly space
 * or a user defined set of discrete frequencies.
 */
unsigned int IMRPhenomHM_is_freq_uniform(
    RealVector *freqs,
    double deltaF)
{
    unsigned int freq_is_uniform = 0;
    if ((freqs->length == 2) && (deltaF > 0.))
    {
        freq_is_uniform = 1;
    }
    else if ((freqs->length != 2) && (deltaF <= 0.))
    {
        freq_is_uniform = 0;
    }

    return freq_is_uniform;
}

/**
 * derive frequency variables for PhenomHM based on input.
 * used to set the index on arrays where we have non-zero values.
 */
int init_IMRPhenomHMGet_FrequencyBounds_storage(
    PhenomHMFrequencyBoundsStorage *p, /**< [out] PhenomHMFrequencyBoundsStorage struct */
    RealVector *freqs,              /**< Input list of GW frequencies [Hz] */
    double Mtot,                        /**< total mass in solar masses */
    double deltaF,                      /**< frequency spacing */
    double f_ref_in                     /**< reference GW frequency */
)
{
    p->deltaF = deltaF;
    /* determine how to populate frequency sequence */
    /* if len(freqs_in) == 2 and deltaF > 0. then
     * f_min = freqs_in[0]
     * f_max = freqs_in[1]
     * else if len(freqs_in) != 2 and deltaF <= 0. then
     * user has given an arbitrary set of frequencies to evaluate the model at.
     */

    p->freq_is_uniform = IMRPhenomHM_is_freq_uniform(freqs, p->deltaF);

    if (p->freq_is_uniform == 1)
    { /* This case we use regularly spaced frequencies */
        p->f_min = freqs->data[0];
        p->f_max = freqs->data[1];

        /* If p->f_max == 0. Then we default to the ending frequency
         * for PhenomHM
         */
        if (p->f_max == 0.)
        {
            p->f_max = PhenomUtilsMftoHz(
                PHENOMHM_DEFAULT_MF_MAX, Mtot);
        }
        /* we only need to evaluate the phase from
         * f_min to f_max with a spacing of deltaF
         */
        p->npts = PhenomInternal_NextPow2(p->f_max / p->deltaF) + 1;
        p->ind_min = (size_t)ceil(p->f_min / p->deltaF);
        p->ind_max = (size_t)ceil(p->f_max / p->deltaF);
        assert ((p->ind_max <= p->npts) && (p->ind_min <= p->ind_max)) ; //, PD_EDOM, "minimum freq index %zu and maximum freq index %zu do not fulfill 0<=ind_min<=ind_max<=npts=%zu.");
    }
    else if (p->freq_is_uniform == 0)
    { /* This case we possibly use irregularly spaced frequencies */
        /* Check that the frequencies are always increasing */
        /*for (unsigned int i = 0; i < freqs->length - 1; i++)
        {
            assert (freqs->data[i] - freqs->data[i + 1] < 0.) ; //,
                //PD_EFUNC,
                //"custom frequencies must be increasing.");
        }*/

        //printf("Using custom frequency input.\n");
        p->f_min = freqs->data[0];
        p->f_max = freqs->data[freqs->length - 1]; /* Last element */

        p->npts = freqs->length;
        p->ind_min = 0;
        p->ind_max = p->npts;
    }
    else
    { /* Throw an informative error. */
        printf("Input sequence of frequencies and deltaF is not \
    compatible.\nSpecify a f_min and f_max by using a RealVector of length = 2 \
    along with a deltaF > 0.\
    \nIf you want to supply an arbitrary list of frequencies to evaluate the with \
    then supply those frequencies using a RealVector and also set deltaF <= 0.");
    }

    /* Fix default behaviour for f_ref */
    /* If f_ref = 0. then set f_ref = f_min */
    p->f_ref = f_ref_in;
    //mkatz correction: want to set f_ref to f_peak of 22 mode (do it in other function)
    /*if (p->f_ref == 0.)
    {
        p->f_ref = p->f_min;
    }*/

    return 1;
}

/**
 * Precompute a bunch of PhenomHM related quantities and store them filling in a
 * PhenomHMStorage variable
 */
static int init_PhenomHM_Storage(
    PhenomHMStorage *p,
    const double m1_SI,
    const double m2_SI,
    const double chi1z,
    const double chi2z,
    RealVector *freqs,
    const double deltaF,
    const double f_ref,
    const double phiRef)
{
    int retcode;
    assert (0 != p) ;

    p->m1 = m1_SI / MSUN_SI;
    p->m2 = m2_SI / MSUN_SI;
    p->m1_SI = m1_SI;
    p->m2_SI = m2_SI;
    p->Mtot = p->m1 + p->m2;
    p->eta = p->m1 * p->m2 / (p->Mtot * p->Mtot);
    p->chi1z = chi1z;
    p->chi2z = chi2z;
    p->phiRef = phiRef;
    p->deltaF = deltaF;
    p->freqs = freqs;

    if (p->eta > 0.25)
        assert(0); //PhenomInternal_nudge(&(p->eta), 0.25, 1e-6);
    if (p->eta > 0.25 || p->eta < 0.0)
        assert(0); //ERROR(PD_EDOM, "Unphysical eta. Must be between 0. and 0.25\n");
    if (p->eta < MAX_ALLOWED_ETA)
        printf("Warning: The model is not calibrated for mass-ratios above 20\n");

    retcode = 0;
    retcode = PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
        &(p->m1),
        &(p->m2),
        &(p->chi1z),
        &(p->chi2z));
    assert (1 == retcode) ; //,
        //PD_EFUNC,
        //"PhenomInternal_AlignedSpinEnforcePrimaryIsm1 failed");

    /* sanity checks on frequencies */
    PhenomHMFrequencyBoundsStorage pHMFS;
    retcode = 0;
    retcode = init_IMRPhenomHMGet_FrequencyBounds_storage(
        &pHMFS,
        p->freqs,
        p->Mtot,
        p->deltaF,
        f_ref);
    assert (1 == retcode) ; //,
        //PD_EFUNC,
        //"init_IMRPhenomHMGet_FrequencyBounds_storage failed");

    /* redundent storage */
    p->f_min = pHMFS.f_min;
    p->f_max = pHMFS.f_max;
    p->f_ref = pHMFS.f_ref;
    p->freq_is_uniform = pHMFS.freq_is_uniform;
    p->npts = pHMFS.npts;
    p->ind_min = pHMFS.ind_min;
    p->ind_max = pHMFS.ind_max;

    p->Mf_ref = PhenomUtilsHztoMf(p->f_ref, p->Mtot);

    p->finmass = IMRPhenomDFinalMass(p->m1, p->m2, p->chi1z, p->chi2z);
    p->finspin = IMRPhenomDFinalSpin(p->m1, p->m2, p->chi1z, p->chi2z); /* dimensionless final spin */
    if (p->finspin > 1.0)
        assert(0); //ERROR(PD_EDOM, "PhenomD fring function: final spin > 1.0 not supported\n");

    /* populate the ringdown frequency array */
    /* If you want to model a new mode then you have to add it here. */
    /* (l,m) = (2,2) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[2][2],
        &p->PhenomHMfdamp[2][2],
        2, 2,
        p->finmass, p->finspin);

    /* (l,m) = (2,1) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[2][1],
        &p->PhenomHMfdamp[2][1],
        2, 1,
        p->finmass, p->finspin);

    /* (l,m) = (3,3) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[3][3],
        &p->PhenomHMfdamp[3][3],
        3, 3,
        p->finmass, p->finspin);

    /* (l,m) = (3,2) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[3][2],
        &p->PhenomHMfdamp[3][2],
        3, 2,
        p->finmass, p->finspin);

    /* (l,m) = (4,4) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[4][4],
        &p->PhenomHMfdamp[4][4],
        4, 4,
        p->finmass, p->finspin);

    /* (l,m) = (4,3) */
    IMRPhenomHMGetRingdownFrequency(
        &p->PhenomHMfring[4][3],
        &p->PhenomHMfdamp[4][3],
        4, 3,
        p->finmass, p->finspin);

    p->Mf_RD_22 = p->PhenomHMfring[2][2];
    p->Mf_DM_22 = p->PhenomHMfdamp[2][2];

    /* (l,m) = (2,2) */
    int ell, mm;
    ell = 2;
    mm = 2;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (2,1) */
    ell = 2;
    mm = 1;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (3,3) */
    ell = 3;
    mm = 3;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (3,2) */
    ell = 3;
    mm = 2;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (4,4) */
    ell = 4;
    mm = 4;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;
    /* (l,m) = (4,3) */
    ell = 4;
    mm = 3;
    p->Rholm[ell][mm] = p->Mf_RD_22 / p->PhenomHMfring[ell][mm];
    p->Taulm[ell][mm] = p->PhenomHMfdamp[ell][mm] / p->Mf_DM_22;

    return 1;
};

/**
 * domain mapping function - ringdown
 */
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
        double Rholm = pHM->Rholm[ell][mm];
        ans = Rholm * Mf; /* Used for the Phase */
    }

    return ans;
}

/**
 * mathematica function Ti
 * domain mapping function - inspiral
 */
double IMRPhenomHMTi(double Mf, const int mm)
{
    return 2.0 * Mf / mm;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
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
    assert (a != NULL) ; //, PD_EFAULT, "a error");
    assert (b != NULL) ; //, PD_EFAULT, "b error");
    assert (fi != NULL) ; //, PD_EFAULT, "fi error");
    assert (fr != NULL) ; //, PD_EFAULT, "fr error");
    assert (f1 != NULL) ; //, PD_EFAULT, "f1 error");

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
    if (ret != 1)
    {
        printf("IMRPhenomHMMapParams failed in IMRPhenomHMFreqDomainMapParams (1)\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }

    return 1;
}

/**
 * IMRPhenomHMFreqDomainMap
 * Input waveform frequency in Geometric units (Mflm)
 * and computes what frequency this corresponds
 * to scaled to the 22 mode.
 */
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
    if (ret != 1)
    {
        printf("IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMap\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }
    double Mf22 = a * Mflm + b;
    return Mf22;
}

int IMRPhenomHMPhasePreComp(
    HMPhasePreComp *q,          /**< [out] HMPhasePreComp struct */
    const int ell,             /**< ell spherical harmonic number */
    const int mm,              /**< m spherical harmonic number */
    PhenomHMStorage *pHM       /**< PhenomHMStorage struct */
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
    if (ret != 1)
    {
        printf("IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMapParams - inspiral\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }
    q->ai = ai;
    q->bi = bi;

    ret = IMRPhenomHMFreqDomainMapParams(&am, &bm, &fi, &fr, &f1, fi + Mfshift, ell, mm, pHM, AmpFlag);
    if (ret != 1)
    {
        printf("IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMapParams - intermediate\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }
    q->am = am;
    q->bm = bm;

    ret = IMRPhenomHMFreqDomainMapParams(&ar, &br, &fi, &fr, &f1, fr + Mfshift, ell, mm, pHM, AmpFlag);
    if (ret != 1)
    {
        printf("IMRPhenomHMFreqDomainMapParams failed in IMRPhenomHMFreqDomainMapParams - merger-ringdown\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }

    q->ar = ar;
    q->br = br;

    q->fi = fi;
    q->fr = fr;

    double Rholm = pHM->Rholm[ell][mm];
    double Taulm = pHM->Taulm[ell][mm];

    PhenDAmpAndPhasePreComp pDPreComp;
    ret = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        Rholm,
        Taulm);
    if (ret != 1)
    {
        printf("IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }

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
    cmplx Hlm = 0;
    double M_INPUT = M1 + M2;
    M1 = M1 / (M_INPUT);
    M2 = M2 / (M_INPUT);
    double M = M1 + M2;
    double eta = M1 * M2 / (M * M);
    double delta = sqrt(1.0 - 4 * eta);
    double Xs = 0.5 * (X1z + X2z);
    double Xa = 0.5 * (X1z - X2z);
    double ans = 0;
    cmplx I(0.0, 1.0);

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
    ans = M * M * PI * sqrt(eta * 2.0 / 3) * pow(v, -3.5) * std::abs(Hlm);

    return ans;
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
 * Returns h+ and hx in the frequency domain.
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
 int IMRPhenomHM(
     ModeContainer *mode_vals, /**< [out] Frequency-domain waveform hx */
     double *freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
     int f_length,
     double m1_SI,                        /**< mass of companion 1 (kg) */
     double m2_SI,                        /**< mass of companion 2 (kg) */
     double chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
     double chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
     const double distance,               /**< distance of source (m) */
     const double phiRef,                 /**< reference orbital phase (rad) */
     const double deltaF,                 /**< Sampling frequency (Hz). To use arbitrary frequency points set deltaF <= 0. */
     double f_ref,                        /**< Reference frequency */
     int num_modes,
     int to_gpu
)
{
    /* define and init return code for this function */
    int retcode;

    /* sanity checks on input parameters: check pointers, etc. */
    /* NOTE: a lot of checks are done in the function
     * IMRPhenomHMGethlmModes because that can also be used
     * as a standalone function. It gets called through IMRPhenomHMCore
     * so to avoid doubling up on checks alot of the checks are done in
     * IMRPhenomHMGethlmModes.
     */
    assert (distance > 0); //, PD_EDOM, "distance must be positive.\n");

    // DECLARE ALL THE  NECESSARY STRUCTS FOR THE GPU
    PhenomHMStorage *pHM_trans = (PhenomHMStorage *) malloc(sizeof(PhenomHMStorage));
    IMRPhenomDAmplitudeCoefficients *pAmp_trans = (IMRPhenomDAmplitudeCoefficients*)malloc(sizeof(IMRPhenomDAmplitudeCoefficients));
    AmpInsPrefactors *amp_prefactors_trans = (AmpInsPrefactors*)malloc(sizeof(AmpInsPrefactors));
    PhenDAmpAndPhasePreComp *pDPreComp_all_trans = (PhenDAmpAndPhasePreComp*)malloc(num_modes*sizeof(PhenDAmpAndPhasePreComp));
    HMPhasePreComp *q_all_trans = (HMPhasePreComp*)malloc(num_modes*sizeof(HMPhasePreComp));
    double t0;
    double phi0;
    double amp0;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        f_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        phiRef,
        deltaF,
        f_ref,
        num_modes,
        to_gpu,
        pHM_trans,
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in IMRPhenomHM.");

    free(pHM_trans);
    free(pAmp_trans);
    free(amp_prefactors_trans);
    free(pDPreComp_all_trans);
    free(q_all_trans);
    /* cleanup */
    /* XLALDestroy and XLALFree any pointers. */

    return 1;
}

/** @} */
/** @} */

/**
 * internal function that returns h+ and hx.
 * Inside this function the my bulk of the work is done
 * like the loop over frequencies.
 */
 void host_calculate_all_modes(ModeContainer *mode_vals, PhenomHMStorage *pHM, double *freqs, double M_tot_sec, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp *pDPreComp_all, HMPhasePreComp *q_all, double amp0, int num_modes, double t0, double phi0){
     unsigned int mm, ell;
     double Rholm, Taulm;
     double freq_geom;
     for (int mode_i=0; mode_i<num_modes; mode_i++)
     {
         ell = mode_vals[mode_i].l;
         mm = mode_vals[mode_i].m;
         Rholm = pHM->Rholm[ell][mm];
         Taulm = pHM->Taulm[ell][mm];
         if ((ell==0) && (mm=0))
             continue;
        for (unsigned int i = pHM->ind_min; i < pHM->ind_max; i++)
        {
            freq_geom = freqs[i]*M_tot_sec;
            host_calculate_each_mode(i, mode_vals[mode_i], ell, mm, pHM, freq_geom, pAmp, amp_prefactors, pDPreComp_all[mode_i], q_all[mode_i], amp0, Rholm, Taulm, t0, phi0);
        }
     }
 }


 void host_calculate_each_mode(int i, ModeContainer mode_val, unsigned int ell, unsigned int mm, PhenomHMStorage *pHM, double freq_geom, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double amp0, double Rholm, double Taulm, double t0, double phi0){
         double freq_amp, Mf, beta_term1, beta, beta_term2, HMamp_term1, HMamp_term2;
         double Mf_wf, Mfr, tmpphaseC, phase_term1, phase_term2;
         double amp_i, phase_i;
         int status_in_for;
         UsefulPowers powers_of_f;
         int retcode = 0;

          /* loop over only positive m is intentional. negative m added automatically */
          // generate amplitude
          // IMRPhenomHMAmplitude
        freq_amp = IMRPhenomHMFreqDomainMap(freq_geom, ell, mm, pHM, AmpFlagTrue);

            status_in_for = 1;
          /* Now generate the waveform */
              Mf = freq_amp; //freqs->data[i]; // geometric frequency

              status_in_for = init_useful_powers(&powers_of_f, Mf);
              if (1 != status_in_for)
              {
                printf("init_useful_powers failed for Mf, status_in_for=%d", status_in_for);
                retcode = status_in_for;
                assert(0); //(0);
              }
              else
              {
                amp_i = IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, &amp_prefactors);
              }


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
            phase_i = cShift[mm];
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
                printf("ERROR - should not get here - in function IMRPhenomHMPhase");
                assert(0); //ERROR(PD_EDOM, "error");
            }
        //}

            //phase_term1 = 0.0;
            //phase_term2 = 0.0;
            //Mf = 0.0;
            Mf = freq_geom;
            phase_term1 = - t0 * (Mf - pHM->Mf_ref);
            phase_term2 = phase_i - (mm * phi0);

            mode_val.phase[i] = phase_term1 + phase_term2;

             /*hlm = amp_i * std::exp(-I * (phase_term1 + phase_term2));
             //double complexFrequencySeries *hlm = XLALSphHarmFrequencySeriesGetMode(*hlms, ell, mm);
             if ((std::real(hlm) == 0.0) && (std::imag(hlm) == 0.0))
             {
                 hptilde->data[mode_i*hptilde->length + i] = 0.0; //TODO check += here
                hctilde->data[mode_i*hctilde->length + i] = 0.0;
             }
             else
             {
                 hptilde->data[mode_i*hptilde->length + i] = factorp * hlm * amp0; //TODO check += here
                 hctilde->data[mode_i*hctilde->length + i] = factorc * hlm * amp0;
             }                */
             //IMRPhenomHMFDAddMode(*hptilde, *hctilde, hlm, inclination, 0., ell, mm, sym); /* The phase \Phi is set to 0 - assumes phiRef is defined as half the phase of the 22 mode h22 */

             //if (mode_i == 1)
             //    printf("%d, %d %e\n", ell, mm, (*hptilde)->data[mode_i*(*hptilde)->length + i]);
         //printf("(l, m): (%d, %d)\n", ell, mm);
}



int IMRPhenomHMCore(
    ModeContainer *mode_vals, /**< [out] Frequency domain hx GW strain */
    double *freqs_trans,                      /**< GW frequecny list [Hz] */
    int f_length,
    double m1_SI,                               /**< primary mass [kg] */
    double m2_SI,                               /**< secondary mass [kg] */
    double chi1z,                               /**< aligned spin of primary */
    double chi2z,                               /**< aligned spin of secondary */
    const double distance,                      /**< distance [m] */
    const double phiRef,                        /**< orbital phase at f_ref */
    const double deltaF,                        /**< frequency spacing */
    double f_ref,                               /**< reference GW frequency */
    int num_modes,
    int to_gpu,
    PhenomHMStorage *pHM_trans,
    IMRPhenomDAmplitudeCoefficients *pAmp_trans,
    AmpInsPrefactors *amp_prefactors_trans,
    PhenDAmpAndPhasePreComp *pDPreComp_all_trans,
    HMPhasePreComp *q_all_trans,
    double *t0,
    double *phi0,
    double *amp0
)
{
    int retcode;
    long ligotimegps_zero = 0;
    int sym;

    // TODO make this more efficient?
    RealVector *freqs = CreateRealVector(f_length);
    freqs->data = freqs_trans;

    /* setup PhenomHM model storage struct / structs */
    /* Compute quantities/parameters related to PhenomD only once and store them */
    //PhenomHMStorage *pHM;
    PhenomHMStorage * pHM = pHM_trans;
    retcode = 0;
    retcode = init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        freqs,
        deltaF,
        f_ref,
        phiRef);
    assert (1 == retcode); //, PD_EFUNC, "init_PhenomHM_Storage \
failed");

    /* setup frequency sequency */
    //RealVector *amps = NULL;
    //RealVector *phases = NULL;
    //RealVector *freqs_geom = NULL; /* freqs is in geometric units */


    /* Two possibilities */
    if (pHM->freq_is_uniform == 1)
    { /* 1. uniformly spaced */
        //printf("freq_is_uniform = True\n");
/**
        freqs = CreateRealVector(pHM->npts);
        phases = CreateRealVector(pHM->npts);
        amps = CreateRealVector(pHM->npts);

        for (size_t i = 0; i < pHM->npts; i++)
        {                                     /* populate the frequency unitformly from zero - this is the standard
             convention we use when generating waveforms in LAL. */
        /**    freqs->data[i] = i * pHM->deltaF; /* This is in Hz */
        //    phases->data[i] = 0;              /* initalise all phases to zero. */
        //    amps->data[i] = 0;                /* initalise all amps to zero. */
        //}
        /* coalesce at t=0 */
        /*CHECK(
            XLALGPSAdd(&tC, -1. / pHM->deltaF),
            PD_EFUNC,
            "Failed to shift coalescence time to t=0,\
tried to apply shift of -1.0/deltaF with deltaF=%g.",
            pHM->deltaF);*/
        ligotimegps_zero += -1. / deltaF;
    }
    else if (pHM->freq_is_uniform == 0)
    { /* 2. arbitrarily space */
        //printf("freq_is_uniform = False\n");
        freqs = pHM->freqs; /* This is in Hz */
        //phases = CreateRealVector(freqs->length);
        //amps = CreateRealVector(freqs->length);
        //for (size_t i = 0; i < pHM->npts; i++)
        //{
        //    phases->data[i] = 0; /* initalise all phases to zero. */
        //    amps->data[i] = 0;   /* initalise all phases to zero. */
        //}
    }
    else
    {
        assert(0); // ERROR(PD_EDOM, "freq_is_uniform is not 0 or 1.");
    }

    /* PhenomD functions take geometric frequencies */
    /*freqs_geom = CreateRealVector(pHM->npts);
    freqs_geom->data = freqs_geom_trans;
    for (size_t i = 0; i < pHM->npts; i++)
    {
        freqs_geom->data[i] = PhenomUtilsHztoMf(freqs->data[i], pHM->Mtot); //1 initalise all phases to zero.
    }
    */

    PhenDAmpAndPhasePreComp pDPreComp22;
    retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp22,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rholm[2][2],
        pHM->Taulm[2][2]);
    if (retcode != 1)
    {
        printf("IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
        assert(0); //ERROR(PD_EDOM, "error");
    }

    if (pHM->f_ref == 0.0){
        pHM->f_ref = pDPreComp22.pAmp.fmaxCalc;
        pHM->Mf_ref = PhenomUtilsHztoMf(pHM->f_ref, pHM->Mtot);
    }

    /* compute the reference phase shift need to align the waveform so that
     the phase is equal to phiRef at the reference frequency f_ref. */
    /* the phase shift is computed by evaluating the phase of the
    (l,m)=(2,2) mode.
    phi0 is the correction we need to add to each mode. */
    double phiRef_to_zero = 0.0;
    double phi_22_at_f_ref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,  1.0, 1.0);
    *phi0 = 0.5 * (phi_22_at_f_ref + phiRef_to_zero); // TODO: check this, I think it should be half of phiRef as well

    *t0 = IMRPhenomDComputet0(
    pHM->eta, pHM->chi1z, pHM->chi2z,
    pHM->finspin);


    retcode = ComputeIMRPhenomDAmplitudeCoefficients_sub(pAmp_trans, pHM->eta, pHM->chi1z, pHM->chi2z,
    pHM->finspin);
    assert(retcode == 1);
    IMRPhenomDAmplitudeCoefficients * pAmp = pAmp_trans;
    if (!pAmp)
      assert(0); //ERROR(PD_EFUNC, "pAmp Failed");

    retcode = 0;
    AmpInsPrefactors *amp_prefactors = amp_prefactors_trans;
    retcode = init_amp_ins_prefactors(amp_prefactors_trans, pAmp);
    assert (1 == retcode); //, retcode, "init_amp_ins_prefactors failed");
    /* compute the frequency bounds */
    const double Mtot = (m1_SI + m2_SI) / MSUN_SI;
    PhenomHMFrequencyBoundsStorage *pHMFS;
    pHMFS = (PhenomHMFrequencyBoundsStorage*)  malloc(sizeof(PhenomHMFrequencyBoundsStorage));
    retcode = 0;
    retcode = init_IMRPhenomHMGet_FrequencyBounds_storage(
        pHMFS,
        freqs,
        Mtot,
        deltaF,
        f_ref);
    assert (1 == retcode); //,
                //PD_EFUNC, "init_IMRPhenomHMGet_FrequencyBounds_storage failed");
   /* Compute the amplitude pre-factor */
   *amp0 = PhenomUtilsFDamp0(Mtot, distance); // TODO check if this is right units

    //HMPhasePreComp q;
    HMPhasePreComp * q_all = q_all_trans;

    PhenDAmpAndPhasePreComp *pDPreComp_all = pDPreComp_all_trans;

    double Rholm, Taulm;
    unsigned int ell, mm;

    for (int mode_i=0; mode_i<num_modes; mode_i++){
        ell = mode_vals[mode_i].l;
        mm = mode_vals[mode_i].m;


        retcode = 0;
        retcode = IMRPhenomHMPhasePreComp(&q_all[mode_i], ell, mm, pHM);
        if (retcode != 1)
        {
            printf("IMRPhenomHMPhasePreComp failed\n");
            assert(0); //ERROR(PD_EDOM, "IMRPhenomHMPhasePreComp failed");
        }
        Rholm = pHM->Rholm[ell][mm];
        Taulm = pHM->Taulm[ell][mm];
        retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
            &pDPreComp_all[mode_i],
            pHM->m1,
            pHM->m2,
            pHM->chi1z,
            pHM->chi2z,
            Rholm,
            Taulm);
        if (retcode != 1)
        {
            printf("IMRPhenomDSetupAmpAndPhaseCoefficients failed\n");
            assert(0); //ERROR(PD_EDOM, "IMRPhenomDSetupAmpAndPhaseCoefficients failed");
        }
    }
    pHM->nmodes = num_modes;

    double M_tot_sec = (pHM->m1 + pHM->m2)*MTSUN_SI;

    if (to_gpu == 0){
        host_calculate_all_modes(mode_vals, pHM, freqs_trans, M_tot_sec, pAmp, *amp_prefactors, pDPreComp_all, q_all, *amp0, num_modes, *t0, *phi0);
    }


    /* Two possibilities */
    if (pHM->freq_is_uniform == 1)
    { /* 1. uniformly spaced */
        //DestroyRealVector(freqs);
        //DestroyRealVector(amps);
        //DestroyRealVector(phases);

    }
    else if (pHM->freq_is_uniform == 0)
    { /* 2. arbitrarily space */
        //DestroyRealVector(freqs);
        //DestroyRealVector(amps);
        //DestroyRealVector(phases);
    }
    else
    {
        assert(0); //ERROR(PD_EDOM, "freq_is_uniform should be either 0 or 1.");
    }
    //free(pAmp);
    //free(pHM);


    //printf("\n\n\n\n\n\n");
    //DestroyRealVector(freqs);
    free(pHMFS);
    //DestroyRealVector(freqs_geom);
    free(freqs); // TODO check this
    return 1;
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

    int to_gpu = 0;
    int to_interp = 0;
    ModeContainer * mode_vals = cpu_create_modes(num_modes, l, m, f_length, to_gpu, to_interp);
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
