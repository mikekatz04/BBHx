/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown in `PhenomHM.cpp`. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.
 */

#ifndef _IMR_PHENOMD_INTERNALS_H
#define _IMR_PHENOMD_INTERNALS_H

/*
 * Copyright (C) 2015 Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
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

// LAL independent code (C) 2017 Michael Puerrer
// merged definitions from IMRPhenomD.h into here.

/**
 * \author Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London
 *
 * \file
 *
 * \brief Internal function for IMRPhenomD phenomenological waveform model.
 * See \ref LALSimIMRPhenom_c for more details.
 *
 */

/*
This waveform uses the TaylorF2 coefficients for it's inspiral phase augmented
by higher order phenomenological terms tuned to SEOBv2-Hybrid waveforms.
Below are lines copied from LALSimInspiralPNCoefficients.c which are the TaylorF2
phase coefficients we have used.
We document them here in case changes to that file changes the behaviour
of this waveform.

    const double mtot = m1 + m2;
    const double d = (m1 - m2) / (m1 + m2);
    const double eta = m1*m2/mtot/mtot;
    const double m1M = m1/mtot;
    const double m2M = m2/mtot;
    // Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
    // We write dSigmaL for their (\delta m/m) * \Sigma_\ell
    // There's a division by mtotal^2 in both the energy and flux terms
    // We just absorb the division by mtotal^2 into SL and dSigmaL

    const double SL = m1M*m1M*chi1L + m2M*m2M*chi2L;
    const double dSigmaL = d*(m2M*chi2L - m1M*chi1L);

    const double pfaN = 3.L/(128.L * eta);
    //Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18
    pfa->v[0] = 1.L;
    pfa->v[2] = 5.L*(743.L/84.L + 11.L * eta)/9.L;
    pfa->v[3] = -16.L*PI;
    pfa->v[4] = 5.L*(3058.673L/7.056L + 5429.L/7.L * eta
                     + 617.L * eta*eta)/72.L;
    pfa->v[5] = 5.L/9.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->vlogv[5] = 5.L/3.L * (7729.L/84.L - 13.L * eta) * PI;
    pfa->v[6] = (11583.231236531L/4.694215680L
                     - 640.L/3.L * PI * PI - 6848.L/21.L*GAMMA)
                 + eta * (-15737.765635L/3.048192L
                     + 2255./12. * PI * PI)
                 + eta*eta * 76055.L/1728.L
                 - eta*eta*eta * 127825.L/1296.L;
    pfa->v[6] += (-6848.L/21.L)*log(4.);
    pfa->vlogv[6] = -6848.L/21.L;
    pfa->v[7] = PI * ( 77096675.L/254016.L
                     + 378515.L/1512.L * eta - 74045.L/756.L * eta*eta);

    // Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16
    const double pn_gamma = (554345.L/1134.L + 110.L*eta/9.L)*SL + (13915.L/84.L - 10.L*eta/3.)*dSigmaL;
    switch( spinO )
    {
        case LAL_SIM_INSPIRAL_SPIN_ORDER_ALL:
        case LAL_SIM_INSPIRAL_SPIN_ORDER_35PN:
            pfa->v[7] += (-8980424995.L/762048.L + 6586595.L*eta/756.L - 305.L*eta*eta/36.L)*SL - (170978035.L/48384.L - 2876425.L*eta/672.L - 4735.L*eta*eta/144.L) * dSigmaL;
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex>

#include <stdbool.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include "globalPhenomHM.h"

// From std/LALConstants.h

#define PI        3.141592653589793238462643383279502884
#define PI_4      0.785398163397448309615660845819875721
#define GAMMA     0.577215664901532860606512090082402431

/**
 * @brief Solar mass, kg
 * @details
 * MSUN_SI = LAL_GMSUN_SI / LAL_G_SI
 */
//#define MSUN_SI 1.988546954961461467461011951140572744e30

/**
 * @brief Geometrized solar mass, s
 * @details
 * MTSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI * LAL_C_SI)
 */
//#define MTSUN_SI 4.925491025543575903411922162094833998e-6

/**
 * @brief Geometrized solar mass, m
 * @details
 * MRSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI)
 */
//#define MRSUN_SI 1.476625061404649406193430731479084713e3

#define PC_SI 3.085677581491367278913937957796471611e16 /**< Parsec, m */

/* CONSTANTS */

/**
 * Dimensionless frequency (Mf) at which define the end of the waveform
 */
#define f_CUT 0.2


/**
  * Minimal final spin value below which the waveform might behave pathological
  * because the ISCO frequency is too low. For more details, see the review wiki
  * page https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/PhenD_LargeNegativeSpins
  */
#define MIN_FINAL_SPIN -0.717

/**
  * A large mass ratio causes memory over-runs.
  * We test and put the limit an order of magnitude above that of previous waveform models (which were around q=100).
  */
#define MAX_ALLOWED_MASS_RATIO 5000

/**
 *
 * \brief Tabulated Quasi-Normal Mode Information for Ringdown
 *
 * This file contains listed final spin values, and corresponding Quasi-Normal-Mode(QNM) central frequencies and decay rates. The spins are on [-1,1] where values are the dimensionless Kerr parameters S/M^2. The frequencies and decay rates are in units of 1/(s*pi*M), where M is 1. To compare values with tables given at, for example, www.phy.olemiss.edu/~berti/ringdown/, multiply by 2*pi, and note that the decay rate corresponds to the imaginary part of the QNM frequency.
 *
 * Values for spins between -0.994 and 0.994 were sourced from the above website (2014), while qualitatively accurate values for the remaining spins where sourced from the implementation used in arxiv:1404:3197. Both references use the analytic representation of QNMs developed by Leaver in 1986, and for spin values within +-0.994, have identical values within numerical error.
 *
 * */

// NOTE: At the moment we have separate functions for each Phenom coefficient;
// these could be collected together

///////////////////////////////////////////////////////////////////////////////

typedef struct tagCOMPLEX16FrequencySeries {
    cmplx *data;
    char *name;
    long epoch;
    double f0;
    double deltaF;
    // Unit sampleUnits;
    size_t length;
} COMPLEX16FrequencySeries;

COMPLEX16FrequencySeries *CreateCOMPLEX16FrequencySeries(
    char *name,
    long epoch,
    double f0,
    double deltaF,
    size_t length
);

COMPLEX16FrequencySeries *ResizeCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs, size_t length);
void DestroyCOMPLEX16FrequencySeries(COMPLEX16FrequencySeries *fs);

///////////////////////////////////////////////////////////////////////////////
// SM: structure for downsampled FD 22-mode Amp/Phase waveforms

typedef struct tagAmpPhaseFDWaveform {
    double* freq;
    double* amp;
    double* phase;
    size_t length;
} AmpPhaseFDWaveform;

AmpPhaseFDWaveform* CreateAmpPhaseFDWaveform(
    size_t length
);

double* CreateConstantsArray(
    size_t length
);

void DestroyAmpPhaseFDWaveform(AmpPhaseFDWaveform* wf);

extern RealVector* CreateRealVector(
    size_t length
);

extern void DestroyRealVector(RealVector* v);

///////////////////////////////////////////////////////////////////////////////

// Handling errors
//typedef enum {
//    PD_SUCCESS = 0,      /**< PD_SUCCESS return value (not an error number) */
//    PD_FAILURE = -1,     /**< Failure return value (not an error number) */
//    PD_EDOM = 33,        /**< Input domain error */
//    PD_EFAULT = 14,      /**< Invalid pointer */
//    PD_EFUNC = 1024,     /**< Internal function call failed bit: "or" this with existing error number */
//    PD_ENOMEM = 12       /**< Memory allocation error */
//} ERROR_type;


//const char *ErrorString(int code);
//void ERROR(ERROR_type e, const char *errstr);
//void CHECK(bool assertion, ERROR_type e, const char *errstr);
//void PRINT_WARNING(const char *warnstr);

///////////////////////////////////////////////////////////////////////////////

// From LALSimInspiral.h


// From LALSimInspiralWaveformFlags.h
typedef enum tagSpinOrder {
    SPIN_ORDER_0PN  = 0,
    SPIN_ORDER_05PN = 1,
    SPIN_ORDER_1PN  = 2,
    SPIN_ORDER_15PN = 3,
    SPIN_ORDER_2PN  = 4,
    SPIN_ORDER_25PN = 5,
    SPIN_ORDER_3PN  = 6,
    SPIN_ORDER_35PN = 7,
    SPIN_ORDER_ALL  = -1
} SpinOrder;

// From LALSimInspiral.h
int TaylorF2AlignedPhasing(PNPhasingSeries **pfa, const double m1, const double m2, const double chi1, const double chi2);
// From LALSimInspiralPNCoefficients.c
void PNPhasing_F2(
	PNPhasingSeries *pfa, /**< \todo UNDOCUMENTED */
	const double m1, /**< Mass of body 1, in Msol */
	const double m2, /**< Mass of body 2, in Msol */
	const double chi1L, /**< Component of dimensionless spin 1 along Lhat */
	const double chi2L, /**< Component of dimensionless spin 2 along Lhat */
	const double chi1sq,/**< Magnitude of dimensionless spin 1 */
	const double chi2sq, /**< Magnitude of dimensionless spin 2 */
	const double chi1dotchi2 /**< Dot product of dimensionles spin 1 and spin 2 */
);

///////////////////////////////////////////////////////////////////////////////

/**
  * Structure holding all coefficients for the amplitude
  */
/**
  * Structure holding all coefficients for the phase
  */

 /**
   * Structure holding all additional coefficients needed for the delta amplitude functions.
   */
typedef struct tagdeltaUtility {
  double f12;
  double f13;
  double f14;
  double f15;
  double f22;
  double f23;
  double f24;
  double f32;
  double f33;
  double f34;
  double f35;
} DeltaUtility;

/*
 *
 * Internal function prototypes; f stands for geometric frequency "Mf"
 *
 */

////////////////////////////// Miscellaneous functions //////////////////////////////

double chiPN(double eta, double chi1, double chi2);
size_t NextPow2(const size_t n);
bool StepFunc_boolean(const double t, const double t1);

CUDA_CALLABLE_MEMBER
 double pow_2_of(double number);
CUDA_CALLABLE_MEMBER
 double pow_3_of(double number);
CUDA_CALLABLE_MEMBER
 double pow_4_of(double number);

double Subtract3PNSS(double m1, double m2, double M, double eta, double chi1, double chi2);

/******************************* Constants to save floating-point pow calculations *******************************/

/**
 * useful powers in GW waveforms: 1/6, 1/3, 2/3, 4/3, 5/3, 7/3, 8/3
 * calculated using only one invocation of 'pow', the rest are just multiplications and divisions
 */

/**
 * must be called before the first usage of *p
 */
CUDA_CALLABLE_MEMBER
int init_useful_powers(UsefulPowers *p, double number);

/**
 * useful powers of PI, calculated once and kept constant - to be initied with a call to
 * init_useful_powers(&powers_of_pi, PI);
 *
 * only declared here, defined in LALSIMIMRPhenomD.c (because this c file is "included" like an h file)
 */
extern UsefulPowers powers_of_pi;

/**
 * used to cache the recurring (frequency-independant) prefactors of AmpInsAnsatz. Must be inited with a call to
 * init_amp_ins_prefactors(&prefactors, p);
 */


/**
 * must be called before the first usage of *prefactors
 */
int init_amp_ins_prefactors(AmpInsPrefactors *prefactors, IMRPhenomDAmplitudeCoefficients* p);

/**
 * used to cache the recurring (frequency-independant) prefactors of PhiInsAnsatzInt. Must be inited with a call to
 * init_phi_ins_prefactors(&prefactors, p, pn);
 */

/**
 * must be called before the first usage of *prefactors
 */
int init_phi_ins_prefactors(PhiInsPrefactors *prefactors, IMRPhenomDPhaseCoefficients* p, PNPhasingSeries *pn);



//////////////////////// Final spin, final mass, fring, fdamp ///////////////////////

double FinalSpin0815_s(double eta, double s);
double FinalSpin0815(double eta, double chi1, double chi2);
double EradRational0815_s(double eta, double s);
double EradRational0815(double eta, double chi1, double chi2);
double fring(double eta, double chi1, double chi2, double finalspin);
double fdamp(double eta, double chi1, double chi2, double finalspin);

/******************************* Amplitude functions *******************************/

double amp0Func(double eta);

///////////////////////////// Amplitude: Inspiral functions /////////////////////////

double rho1_fun(double eta, double chiPN);
double rho2_fun(double eta, double chiPN);
double rho3_fun(double eta, double chiPN);

CUDA_CALLABLE_MEMBER
double AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors);
double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p);

////////////////////////// Amplitude: Merger-Ringdown functions //////////////////////

double gamma1_fun(double eta, double chiPN);
double gamma2_fun(double eta, double chiPN);
double gamma3_fun(double eta, double chiPN);

CUDA_CALLABLE_MEMBER
double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
double fmaxCalc(IMRPhenomDAmplitudeCoefficients* p);

//////////////////////////// Amplitude: Intermediate functions ///////////////////////

CUDA_CALLABLE_MEMBER
double AmpIntAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p);
double AmpIntColFitCoeff(double eta, double chiPN); //this is the v2 value
double delta0_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
double delta1_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
double delta2_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
double delta3_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
double delta4_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d);
void ComputeDeltasFromCollocation(IMRPhenomDAmplitudeCoefficients* p);

///////////////////////////// Amplitude: glueing function ////////////////////////////

IMRPhenomDAmplitudeCoefficients* ComputeIMRPhenomDAmplitudeCoefficients(double eta, double chi1, double chi2, double finspin);
int ComputeIMRPhenomDAmplitudeCoefficients_sub(IMRPhenomDAmplitudeCoefficients *p, double eta, double chi1, double chi2, double finspin);
CUDA_CALLABLE_MEMBER
double IMRPhenDAmplitude(double f, IMRPhenomDAmplitudeCoefficients *p, UsefulPowers *powers_of_f, AmpInsPrefactors * prefactors);

/********************************* Phase functions *********************************/

/////////////////////////////// Phase: Ringdown functions ////////////////////////////

double alpha1Fit(double eta, double chiPN);
double alpha2Fit(double eta, double chiPN);
double alpha3Fit(double eta, double chiPN);
double alpha4Fit(double eta, double chiPN);
double alpha5Fit(double eta, double chiPN);
CUDA_CALLABLE_MEMBER
double PhiMRDAnsatzInt(double f, IMRPhenomDPhaseCoefficients *p, double Rholm, double Taulm);
double DPhiMRD(double f, IMRPhenomDPhaseCoefficients *p, double Rholm, double Taulm);

/////////////////////////// Phase: Intermediate functions ///////////////////////////

double beta1Fit(double eta, double chiPN);
double beta2Fit(double eta, double chiPN);
double beta3Fit(double eta, double chiPN);
CUDA_CALLABLE_MEMBER
double PhiIntAnsatz(double f, IMRPhenomDPhaseCoefficients *p);
double DPhiIntAnsatz(double f, IMRPhenomDPhaseCoefficients *p);
double DPhiIntTemp(double ff, IMRPhenomDPhaseCoefficients *p);

///////////////////////////// Phase: Inspiral functions /////////////////////////////

double sigma1Fit(double eta, double chiPN);
double sigma2Fit(double eta, double chiPN);
double sigma3Fit(double eta, double chiPN);
double sigma4Fit(double eta, double chiPN);

CUDA_CALLABLE_MEMBER
double PhiInsAnsatzInt(double f, UsefulPowers * powers_of_Mf, PhiInsPrefactors * prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn);
CUDA_CALLABLE_MEMBER
double DPhiInsAnsatzInt(double ff, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn);

////////////////////////////// Phase: glueing function //////////////////////////////

IMRPhenomDPhaseCoefficients* ComputeIMRPhenomDPhaseCoefficients(double eta, double chi1, double chi2, double finspin);
void ComputeIMRPhenDPhaseConnectionCoefficients(IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, PhiInsPrefactors *prefactors, double Rholm, double Taulm);
CUDA_CALLABLE_MEMBER
double IMRPhenDPhase(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, UsefulPowers *powers_of_f, PhiInsPrefactors *prefactors, double Rholm, double Taulm);
/*
void CHECK(bool assertion, ERROR_type e, const char *errstr);
void ERROR(ERROR_type e, const char *errstr);
void PRINT_WARNING(const char *warnstr);*/

typedef struct tagCOMPLEX2dArray {
    cmplx*    data;
    size_t      length;
    size_t      num_modes;
} COMPLEX2dArray;

cmplx cpolar(double r, double th);
cmplx crect(double re, double im);

COMPLEX2dArray *CreateCOMPLEX2dArray(
    size_t length,
    size_t num_modes
);

void DestroyCOMPLEX2dArray(COMPLEX2dArray *tmp);

/**
 * Convert from geometric frequency to frequency in Hz
 */
double PhenomUtilsMftoHz(
    double Mf,       /**< Geometric frequency */
    double Mtot_Msun /**< Total mass in solar masses */
);

/**
 * Convert from frequency in Hz to geometric frequency
 */
double PhenomUtilsHztoMf(
    double fHz,      /**< Frequency in Hz */
    double Mtot_Msun /**< Total mass in solar masses */
);

/**
 * compute the frequency domain amplitude pre-factor
 */
double PhenomUtilsFDamp0(
    double Mtot_Msun, /**< total mass in solar masses */
    double distance   /**< distance (m) */
);



/**
 * Computes the (s)Y(l,m) spin-weighted spherical harmonic.
 *
 * From somewhere ....
 *
 * See also:
 * Implements Equations (II.9)-(II.13) of
 * D. A. Brown, S. Fairhurst, B. Krishnan, R. A. Mercer, R. K. Kopparapu,
 * L. Santamaria, and J. T. Whelan,
 * "Data formats for numerical relativity waves",
 * arXiv:0709.0093v1 (2007).
 *
 * Currently only supports s=-2, l=2,3,4,5,6,7,8 modes.
 */
cmplx SpinWeightedSphericalHarmonic(
                                   double theta,  /**< polar angle (rad) */
                                   double phi,    /**< azimuthal angle (rad) */
                                   int s,        /**< spin weight */
                                   int l,        /**< mode number l */
                                   int m         /**< mode number m */
    );


bool PhenomInternal_approx_equal(double x, double y, double epsilon);

/**
 * If x and X are approximately equal to relative accuracy epsilon
 * then set x = X.
 * If X = 0 then use an absolute comparison.
 */
void PhenomInternal_nudge(double *x, double X, double epsilon);

/**
 * Return the closest higher power of 2
 */
size_t PhenomInternal_NextPow2(const size_t n);

/**
 * Given m1 with aligned-spin chi1z and m2 with aligned-spin chi2z.
 * Enforce that m1 >= m2 and swap spins accordingly.
 * Enforce that the primary object (heavier) is indexed by 1.
 * To be used with aligned-spin waveform models.
 * TODO: There is another function for precessing waveform models
 */
int PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
    double *m1,    /**< [out] mass of body 1 */
    double *m2,    /**< [out] mass of body 2 */
    double *chi1z, /**< [out] aligned-spin component of body 1 */
    double *chi2z  /**< [out] aligned-spin component of body 2 */
);

IMRPhenomDAmplitudeCoefficients* inspiral_only_ComputeIMRPhenomDAmplitudeCoefficients(double eta, double chi1, double chi2);
IMRPhenomDPhaseCoefficients* inspiral_only_ComputeIMRPhenomDPhaseCoefficients(double eta, double chi1, double chi2);


#endif	// of #ifndef _LALSIM_IMR_PHENOMD_INTERNALS_H
