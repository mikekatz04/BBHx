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
#include <random>

#include <stdbool.h>
#include "full.h"

#include "cusparse_v2.h"

#define  NUM_THREADS 256
#define  NUM_THREADS2 64
#define  NUM_THREADS3 128
#define  NUM_THREADS4 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/////////////////////////

// RINGDOWN

/////////////////////////

/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.

* Copyright (C) 2016 Lionel London
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

/*
* Based on the paper by London and Fauchon-Jones: https://arxiv.org/abs/1810.03550
* Basic NOTE(s):
*   - This file contains a function, CW07102016, which outputs complex valued, UNITLESS, QNM frequencies (i.e. Mw) for various QNMs
*   - Usage: cw = CW07102016( kappa, l, m, n ); where cw = Mw + 1i*M/tau; NOTE that kappa is a function of final spin, l and m
*   - See definition of KAPPA below.
*/

/*
* -------------------------------------------------------------------------------- *
* Low level models: QNM Frequencies
* -------------------------------------------------------------------------------- *
*/

/*
* Domain mapping for dimnesionless BH spin
*/
__device__
double SimRingdownCW_KAPPA(double jf, int l, int m)
{
    /* */
    /* if ( jf > 1.0 ) ERROR(XLAL_EDOM, "Spin (dimensionless Kerr parameter) must not be greater than 1.0\n"); */
    /**/
    double alpha = log(2.0 - jf) / log(3.0);
    double beta = 1.0 / (2.0 + l - abs(m));
    return pow(alpha, beta);
}

/*
* Dimensionless QNM Frequencies: Note that name encodes date of writing
*/
/*TODO: Make the function arg comments compatible with doxygen*/
__device__
cmplx SimRingdownCW_CW07102016(double kappa, /* Domain mapping for  remnant BH's spin (Dimensionless) */
                                        int l,        /* Polar eigenvalue */
                                        int input_m,  /* Azimuthal eigenvalue*/
                                        int n)
{ /* Overtone Number*/

    /* Predefine powers to increase efficiency*/
    double kappa2 = kappa * kappa;
    double kappa3 = kappa2 * kappa;
    double kappa4 = kappa3 * kappa;

    /* NOTE that |m| will be used to determine the fit to use, and if input_m < 0, then a conjugate will be taken*/
    int m = abs(input_m);

    /**/
    cmplx j = cmplx(0.0, 1.0);

    /* Initialize the answer*/
    cmplx ans;

    /* Use If-Else ladder to determine which mode function to evaluate*/
    if (2 == l && 2 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (2,2,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.0 + kappa * (1.557847 * gcmplx::exp(2.903124 * j) +
                             1.95097051 * gcmplx::exp(5.920970 * j) * kappa +
                             2.09971716 * gcmplx::exp(2.760585 * j) * kappa2 +
                             1.41094660 * gcmplx::exp(5.914340 * j) * kappa3 +
                             0.41063923 * gcmplx::exp(2.795235 * j) * kappa4);
    }
    else if (2 == l && 2 == m && 1 == n)
    {

        /* Fit for (l,m,n) == (2,2,1). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.0 + kappa * (1.870939 * gcmplx::exp(2.511247 * j) +
                             2.71924916 * gcmplx::exp(5.424999 * j) * kappa +
                             3.05648030 * gcmplx::exp(2.285698 * j) * kappa2 +
                             2.05309677 * gcmplx::exp(5.486202 * j) * kappa3 +
                             0.59549897 * gcmplx::exp(2.422525 * j) * kappa4);
    }
    else if (3 == l && 2 == m && 0 == n)
    {

        /* Define extra powers as needed*/
        double kappa5 = kappa4 * kappa;
        double kappa6 = kappa5 * kappa;

        /* Fit for (l,m,n) == (3,2,0). This is NOT a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.022464 * gcmplx::exp(0.004870 * j) +
              0.24731213 * gcmplx::exp(0.665292 * j) * kappa +
              1.70468239 * gcmplx::exp(3.138283 * j) * kappa2 +
              0.94604882 * gcmplx::exp(0.163247 * j) * kappa3 +
              1.53189884 * gcmplx::exp(5.703573 * j) * kappa4 +
              2.28052668 * gcmplx::exp(2.685231 * j) * kappa5 +
              0.92150314 * gcmplx::exp(5.841704 * j) * kappa6;
    }
    else if (4 == l && 4 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (4,4,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 2.0 + kappa * (2.658908 * gcmplx::exp(3.002787 * j) +
                             2.97825567 * gcmplx::exp(6.050955 * j) * kappa +
                             3.21842350 * gcmplx::exp(2.877514 * j) * kappa2 +
                             2.12764967 * gcmplx::exp(5.989669 * j) * kappa3 +
                             0.60338186 * gcmplx::exp(2.830031 * j) * kappa4);
    }
    else if (2 == l && 1 == m && 0 == n)
    {

        /* Define extra powers as needed*/
        double kappa5 = kappa4 * kappa;
        double kappa6 = kappa5 * kappa;

        /* Fit for (l,m,n) == (2,1,0). This is NOT a zero-damped mode in the extremal Kerr limit.*/
        ans = 0.589113 * gcmplx::exp(0.043525 * j) +
              0.18896353 * gcmplx::exp(2.289868 * j) * kappa +
              1.15012965 * gcmplx::exp(5.810057 * j) * kappa2 +
              6.04585476 * gcmplx::exp(2.741967 * j) * kappa3 +
              11.12627777 * gcmplx::exp(5.844130 * j) * kappa4 +
              9.34711461 * gcmplx::exp(2.669372 * j) * kappa5 +
              3.03838318 * gcmplx::exp(5.791518 * j) * kappa6;
    }
    else if (3 == l && 3 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (3,3,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (2.095657 * gcmplx::exp(2.964973 * j) +
                             2.46964352 * gcmplx::exp(5.996734 * j) * kappa +
                             2.66552551 * gcmplx::exp(2.817591 * j) * kappa2 +
                             1.75836443 * gcmplx::exp(5.932693 * j) * kappa3 +
                             0.49905688 * gcmplx::exp(2.781658 * j) * kappa4);
    }
    else if (3 == l && 3 == m && 1 == n)
    {

        /* Fit for (l,m,n) == (3,3,1). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (2.339070 * gcmplx::exp(2.649692 * j) +
                             3.13988786 * gcmplx::exp(5.552467 * j) * kappa +
                             3.59156756 * gcmplx::exp(2.347192 * j) * kappa2 +
                             2.44895997 * gcmplx::exp(5.443504 * j) * kappa3 +
                             0.70040804 * gcmplx::exp(2.283046 * j) * kappa4);
    }
    else if (4 == l && 3 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (4,3,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (0.205046 * gcmplx::exp(0.595328 * j) +
                             3.10333396 * gcmplx::exp(3.016200 * j) * kappa +
                             4.23612166 * gcmplx::exp(6.038842 * j) * kappa2 +
                             3.02890198 * gcmplx::exp(2.826239 * j) * kappa3 +
                             0.90843949 * gcmplx::exp(5.915164 * j) * kappa4);
    }
    else if (5 == l && 5 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (5,5,0). This is a zero-damped mode in the extremal Kerr limit. */
        ans = 2.5 + kappa * (3.240455 * gcmplx::exp(3.027869 * j) +
                             3.49056455 * gcmplx::exp(6.088814 * j) * kappa +
                             3.74704093 * gcmplx::exp(2.921153 * j) * kappa2 +
                             2.47252790 * gcmplx::exp(6.036510 * j) * kappa3 +
                             0.69936568 * gcmplx::exp(2.876564 * j) * kappa4);
    }
    else
    {

        /**/
        ans = 0.0;

    } /* END of IF-ELSE Train for QNM cases */

    /* If m<0, then take the *Negative* conjugate */
    if (input_m < 0)
    {
        /**/
        ans = -1.0 * gcmplx::conj(ans);
    }

    return ans;

} /* END of CW07102016 */


////////////////////

// phenomD

////////////////////


/**
 * calc square of number without floating point 'pow'
 */
__device__
double pow_2_of(double number)
{
	return (number*number);
}

/**
 * calc cube of number without floating point 'pow'
 */
 __device__
double pow_3_of(double number)
{
	return (number*number*number);
}

/**
 * calc fourth power of number without floating point 'pow'
 */
__device__
double pow_4_of(double number)
{
	double pow2 = pow_2_of(number);
	return pow2 * pow2;
}



// From LALSimInspiralPNCoefficients.c
/* The phasing function for TaylorF2 frequency-domain waveform.
 * This function is tested in ../test/PNCoefficients.c for consistency
 * with the energy and flux in this file.
 */
 __device__
void PNPhasing_F2(
	PNPhasingSeries *pfa, /**< \todo UNDOCUMENTED */
	const double m1, /**< Mass of body 1, in Msol */
	const double m2, /**< Mass of body 2, in Msol */
	const double chi1L, /**< Component of dimensionless spin 1 along Lhat */
	const double chi2L, /**< Component of dimensionless spin 2 along Lhat */
	const double chi1sq,/**< Magnitude of dimensionless spin 1 */
	const double chi2sq, /**< Magnitude of dimensionless spin 2 */
	const double chi1dotchi2 /**< Dot product of dimensionles spin 1 and spin 2 */
	)
{
    const double mtot = m1 + m2;
    const double d = (m1 - m2) / (m1 + m2);
    const double eta = m1*m2/mtot/mtot;
    const double m1M = m1/mtot;
    const double m2M = m2/mtot;
    /* Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
     * We write dSigmaL for their (\delta m/m) * \Sigma_\ell
     * There's a division by mtotal^2 in both the energy and flux terms
     * We just absorb the division by mtotal^2 into SL and dSigmaL
     */
    const double SL = m1M*m1M*chi1L + m2M*m2M*chi2L;
    const double dSigmaL = d*(m2M*chi2L - m1M*chi1L);

    const double pfaN = 3.L/(128.L * eta);

    /* Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
    pfa->v0 = 1.L * pfaN;
    pfa->v1 = 0.L * pfaN;
    pfa->v2 = pfaN * (5.L*(743.L/84.L + 11.L * eta)/9.L);
    pfa->v3 = pfaN * (-16.L*PI);
    pfa->v4 = pfaN * (5.L*(3058.673L/7.056L + 5429.L/7.L * eta
                     + 617.L * eta*eta)/72.L);
    pfa->v5 = pfaN * (5.L/9.L * (7729.L/84.L - 13.L * eta) * PI);
    pfa->vlogv5 = pfaN * (5.L/3.L * (7729.L/84.L - 13.L * eta) * PI);
    pfa->v6 = pfaN * ((11583.231236531L/4.694215680L
                     - 640.L/3.L * PI * PI - 6848.L/21.L*GAMMA)
                 + eta * (-15737.765635L/3.048192L
                     + 2255./12. * PI * PI)
                 + eta*eta * 76055.L/1728.L
                 - eta*eta*eta * 127825.L/1296.L);
    pfa->v6 += pfaN * (-6848.L/21.L)*log(4.);
    pfa->vlogv6 = pfaN * (-6848.L/21.L);
    pfa->v7 = pfaN * (PI * ( 77096675.L/254016.L
                     + 378515.L/1512.L * eta - 74045.L/756.L * eta*eta));

    double qm_def1=1.0;
    double qm_def2=1.0;

    /* Compute 2.0PN SS, QM, and self-spin */
    // See Eq. (6.24) in arXiv:0810.5336
    // 9b,c,d in arXiv:astro-ph/0504538
    double pn_sigma = eta * (721.L/48.L*chi1L*chi2L - 247.L/48.L*chi1dotchi2);
    pn_sigma += (720.L*qm_def1 - 1.L)/96.0L * m1M * m1M * chi1L * chi1L;
    pn_sigma += (720.L*qm_def2 - 1.L)/96.0L * m2M * m2M * chi2L * chi2L;
    pn_sigma -= (240.L*qm_def1 - 7.L)/96.0L * m1M * m1M * chi1sq;
    pn_sigma -= (240.L*qm_def2 - 7.L)/96.0L * m2M * m2M * chi2sq;

    double pn_ss3 =  (326.75L/1.12L + 557.5L/1.8L*eta)*eta*chi1L*chi2L;
    pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m1M-120.L*m1M*m1M)*qm_def1 + (-4108.25L/6.72L-108.5L/1.2L*m1M+125.5L/3.6L*m1M*m1M)) *m1M*m1M * chi1sq;
    pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m2M-120.L*m2M*m2M)*qm_def2 + (-4108.25L/6.72L-108.5L/1.2L*m2M+125.5L/3.6L*m2M*m2M)) *m2M*m2M * chi2sq;

    /* Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16 */
    const double pn_gamma = (554345.L/1134.L + 110.L*eta/9.L)*SL + (13915.L/84.L - 10.L*eta/3.L)*dSigmaL;
    pfa->v7 += pfaN * ((-8980424995.L/762048.L + 6586595.L*eta/756.L - 305.L*eta*eta/36.L)*SL - (170978035.L/48384.L - 2876425.L*eta/672.L - 4735.L*eta*eta/144.L) * dSigmaL);

}




// From LALSimInspiralTaylorF2.c
/** \brief Returns structure containing TaylorF2 phasing coefficients for given
 *  physical parameters.
 */
 __device__
int TaylorF2AlignedPhasing(
        PNPhasingSeries *pn,   /**< phasing coefficients (output) */
        const double m1,         /**< mass of body 1 */
        const double m2,		/**< mass of body 2 */
        const double chi1,	/**< aligned spin parameter of body 1 */
        const double chi2	/**< aligned spin parameter of body 2 */
	)
{

    PNPhasing_F2(pn, m1, m2, chi1, chi2, chi1*chi1, chi2*chi2, chi1*chi2);

    return 1;
}

///////////////////////////////////////////////////////////////////////////////

/**
 * PN reduced spin parameter
 * See Eq 5.9 in http://arxiv.org/pdf/1107.1267v2.pdf
 */
__device__
double chiPN(double eta, double chi1, double chi2) {
  // Convention m1 >= m2 and chi1 is the spin on m1
  double delta = sqrt(1.0 - 4.0*eta);
  double chi_s = (chi1 + chi2) / 2.0;
  double chi_a = (chi1 - chi2) / 2.0;
  return chi_s * (1.0 - eta*76.0/113.0) + delta*chi_a;
}


__device__
int init_useful_powers(UsefulPowers *p, double number)
{

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

/******************************* Amplitude functions *******************************/

/**
 * amplitude scaling factor defined by eq. 17 in 1508.07253
 */
__device__
double amp0Func(double eta) {
  return (sqrt(2.0/3.0)*sqrt(eta))/pow(PI, 1./6.);
}


///////////////////////////// Amplitude: Inspiral functions /////////////////////////

// Phenom coefficients rho1, ..., rho3 from direct fit
// AmpInsDFFitCoeffChiPNFunc[eta, chiPN]

/**
 * rho_1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double rho1_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 3931.8979897196696 - 17395.758706812805*eta
  + (3132.375545898835 + 343965.86092361377*eta - 1.2162565819981997e6*eta2)*xi
  + (-70698.00600428853 + 1.383907177859705e6*eta - 3.9662761890979446e6*eta2)*xi2
  + (-60017.52423652596 + 803515.1181825735*eta - 2.091710365941658e6*eta2)*xi3;
}

/**
 * rho_2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double rho2_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -40105.47653771657 + 112253.0169706701*eta
  + (23561.696065836168 - 3.476180699403351e6*eta + 1.137593670849482e7*eta2)*xi
  + (754313.1127166454 - 1.308476044625268e7*eta + 3.6444584853928134e7*eta2)*xi2
  + (596226.612472288 - 7.4277901143564405e6*eta + 1.8928977514040343e7*eta2)*xi3;
}

/**
 * rho_3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double rho3_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 83208.35471266537 - 191237.7264145924*eta +
  (-210916.2454782992 + 8.71797508352568e6*eta - 2.6914942420669552e7*eta2)*xi
  + (-1.9889806527362722e6 + 3.0888029960154563e7*eta - 8.390870279256162e7*eta2)*xi2
  + (-1.4535031953446497e6 + 1.7063528990822166e7*eta - 4.2748659731120914e7*eta2)*xi3;
}

// The Newtonian term in LAL is fine and we should use exactly the same (either hardcoded or call).
// We just use the Mathematica expression for convenience.
/**
 * Inspiral amplitude plus rho phenom coefficents. rho coefficients computed
 * in rho1_fun, rho2_fun, rho3_fun functions.
 * Amplitude is a re-expansion. See 1508.07253 and Equation 29, 30 and Appendix B arXiv:1508.07253 for details
 */
__device__
double AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors) {
  double Mf2 = powers_of_Mf->two;
  double Mf3 = Mf*Mf2;

  return 1 + powers_of_Mf->two_thirds * prefactors->two_thirds
			+ Mf * prefactors->one + powers_of_Mf->four_thirds * prefactors->four_thirds
			+ powers_of_Mf->five_thirds * prefactors->five_thirds + Mf2 * prefactors->two
			+ powers_of_Mf->seven_thirds * prefactors->seven_thirds + powers_of_Mf->eight_thirds * prefactors->eight_thirds
			+ Mf3 * prefactors->three;
}

__device__
int init_amp_ins_prefactors(AmpInsPrefactors * prefactors, IMRPhenomDAmplitudeCoefficients* p)
{
	double eta = p->eta;

	prefactors->amp0 = amp0Func(p->eta);

	double chi1 = p->chi1;
	double chi2 = p->chi2;
	double rho1 = p->rho1;
	double rho2 = p->rho2;
	double rho3 = p->rho3;

	double chi12 = chi1*chi1;
	double chi22 = chi2*chi2;
	double eta2 = eta*eta;
	double eta3 = eta*eta2;

    UsefulPowers powers_of_pi;
    init_useful_powers(&powers_of_pi, PI);

	double Pi = PI;
	double Pi2 = powers_of_pi.two;
	double Seta = sqrt(1.0 - 4.0*eta);

	prefactors->two_thirds = ((-969 + 1804*eta)*powers_of_pi.two_thirds)/672.;
	prefactors->one = ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.;
	prefactors->four_thirds = (	(-27312085.0 - 10287648*chi22 - 10287648*chi12*(1 + Seta) + 10287648*chi22*Seta
								 + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
								 + 35371056*eta2
								 )
							* powers_of_pi.four_thirds) / 8.128512e6;
	prefactors->five_thirds = (powers_of_pi.five_thirds * (chi2*(-285197*(-1 + Seta) + 4*(-91902 + 1579*Seta)*eta - 35632*eta2)
															+ chi1*(285197*(1 + Seta) - 4*(91902 + 1579*Seta)*eta - 35632*eta2)
															+ 42840*(-1.0 + 4*eta)*Pi
															)
								) / 32256.;
	prefactors->two = - (Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2
							  - 324322727232*eta3
							  - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta)
									- 107414046432*chi22*Seta + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi
									)
							  + 12*eta*(-545384828789 - 176491177632*chi1*chi2 + 202603761360*chi22
										+ 77616*chi12*(2610335 + 995766*Seta) - 77287373856*chi22*Seta
										+ 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2
										)
								)
						)/6.0085960704e10;
	prefactors->seven_thirds= rho1;
	prefactors->eight_thirds = rho2;
	prefactors->three = rho3;

	return 1;
}

/**
 * Take the AmpInsAnsatz expression and compute the first derivative
 * with respect to frequency to get the expression below.
 */
__device__
double DAmpInsAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double eta = p->eta;
  double chi1 = p->chi1;
  double chi2 = p->chi2;
  double rho1 = p->rho1;
  double rho2 = p->rho2;
  double rho3 = p->rho3;

  double chi12 = chi1*chi1;
  double chi22 = chi2*chi2;
  double eta2 = eta*eta;
  double eta3 = eta*eta2;
  double Mf2 = Mf*Mf;
  double Pi = PI;
  double Pi2 = Pi * Pi;
  double Seta = sqrt(1.0 - 4.0*eta);

   return ((-969 + 1804*eta)*pow(Pi,2.0/3.0))/(1008.*pow(Mf,1.0/3.0))
   + ((chi1*(81*(1 + Seta) - 44*eta) + chi2*(81 - 81*Seta - 44*eta))*Pi)/48.
   + ((-27312085 - 10287648*chi22 - 10287648*chi12*(1 + Seta)
   + 10287648*chi22*Seta + 24*(-1975055 + 857304*chi12 - 994896*chi1*chi2 + 857304*chi22)*eta
   + 35371056*eta2)*pow(Mf,1.0/3.0)*pow(Pi,4.0/3.0))/6.096384e6
   + (5*pow(Mf,2.0/3.0)*pow(Pi,5.0/3.0)*(chi2*(-285197*(-1 + Seta)
   + 4*(-91902 + 1579*Seta)*eta - 35632*eta2) + chi1*(285197*(1 + Seta)
   - 4*(91902 + 1579*Seta)*eta - 35632*eta2) + 42840*(-1 + 4*eta)*Pi))/96768.
   - (Mf*Pi2*(-336*(-3248849057.0 + 2943675504*chi12 - 3339284256*chi1*chi2 + 2943675504*chi22)*eta2 - 324322727232*eta3
   - 7*(-177520268561 + 107414046432*chi22 + 107414046432*chi12*(1 + Seta) - 107414046432*chi22*Seta
   + 11087290368*(chi1 + chi2 + chi1*Seta - chi2*Seta)*Pi)
   + 12*eta*(-545384828789.0 - 176491177632*chi1*chi2 + 202603761360*chi22 + 77616*chi12*(2610335 + 995766*Seta)
   - 77287373856*chi22*Seta + 5841690624*(chi1 + chi2)*Pi + 21384760320*Pi2)))/3.0042980352e10
   + (7.0/3.0)*pow(Mf,4.0/3.0)*rho1 + (8.0/3.0)*pow(Mf,5.0/3.0)*rho2 + 3*Mf2*rho3;
}

/////////////////////////// Amplitude: Merger-Ringdown functions ///////////////////////

// Phenom coefficients gamma1, ..., gamma3
// AmpMRDAnsatzFunc[]

/**
 * gamma 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double gamma1_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.006927402739328343 + 0.03020474290328911*eta
  + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2)*xi
  + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi2
  + (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi3;
}

/**
 * gamma 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double gamma2_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 1.010344404799477 + 0.0008993122007234548*eta
  + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2)*xi
  + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi2
  + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi3;
}

/**
 * gamma 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double gamma3_fun(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 1.3081615607036106 - 0.005537729694807678*eta
  + (-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2)*xi
  + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi2
  + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi3;
}

/**
 * Ansatz for the merger-ringdown amplitude. Equation 19 arXiv:1508.07253
 */
 __device__
double AmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;
  double fDMgamma3 = fDM*gamma3;
  double fminfRD = f - fRD;
  return exp( -(fminfRD)*gamma2 / (fDMgamma3) )
    * (fDMgamma3*gamma1) / (pow_2_of(fminfRD) + pow_2_of(fDMgamma3));
}

/**
 * first frequency derivative of AmpMRDAnsatz
 */
__device__
double DAmpMRDAnsatz(double f, IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma1 = p->gamma1;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  double fDMgamma3 = fDM * gamma3;
  double pow2_fDMgamma3 = pow_2_of(fDMgamma3);
  double fminfRD = f - fRD;
  double expfactor = exp(((fminfRD)*gamma2)/(fDMgamma3));
  double pow2pluspow2 = pow_2_of(fminfRD) + pow2_fDMgamma3;

   return (-2*fDM*(fminfRD)*gamma3*gamma1) / ( expfactor * pow_2_of(pow2pluspow2)) -
     (gamma2*gamma1) / ( expfactor * (pow2pluspow2)) ;
}

/**
 * Equation 20 arXiv:1508.07253 (called f_peak in paper)
 * analytic location of maximum of AmpMRDAnsatz
 */
__device__
double fmaxCalc(IMRPhenomDAmplitudeCoefficients* p) {
  double fRD = p->fRD;
  double fDM = p->fDM;
  double gamma2 = p->gamma2;
  double gamma3 = p->gamma3;

  // NOTE: There's a problem with this expression from the paper becoming imaginary if gamma2>=1
  // Fix: if gamma2 >= 1 then set the square root term to zero.
  if (gamma2 <= 1)
    return fabs(fRD + (fDM*(-1 + sqrt(1 - pow_2_of(gamma2)))*gamma3)/gamma2);
  else
    return fabs(fRD + (fDM*(-1)*gamma3)/gamma2);
}

///////////////////////////// Amplitude: Intermediate functions ////////////////////////

// Phenom coefficients delta0, ..., delta4 determined from collocation method
// (constraining 3 values and 2 derivatives)
// AmpIntAnsatzFunc[]

/**
 * Ansatz for the intermediate amplitude. Equation 21 arXiv:1508.07253
 */
__device__
double AmpIntAnsatz(double Mf, IMRPhenomDAmplitudeCoefficients* p) {
  double Mf2 = Mf*Mf;
  double Mf3 = Mf*Mf2;
  double Mf4 = Mf*Mf3;
  return p->delta0 + p->delta1*Mf + p->delta2*Mf2 + p->delta3*Mf3 + p->delta4*Mf4;
}

/**
 * The function name stands for 'Amplitude Intermediate Collocation Fit Coefficient'
 * This is the 'v2' value in Table 5 of arXiv:1508.07253
 */
__device__
double AmpIntColFitCoeff(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.8149838730507785 + 2.5747553517454658*eta
  + (1.1610198035496786 - 2.3627771785551537*eta + 6.771038707057573*eta2)*xi
  + (0.7570782938606834 - 2.7256896890432474*eta + 7.1140380397149965*eta2)*xi2
  + (0.1766934149293479 - 0.7978690983168183*eta + 2.1162391502005153*eta2)*xi3;
}

  /**
  * The following functions (delta{0,1,2,3,4}_fun) were derived
  * in mathematica according to
  * the constraints detailed in arXiv:1508.07253,
  * section 'Region IIa - intermediate'.
  * These are not given in the paper.
  * Can be rederived by solving Equation 21 for the constraints
  * given in Equations 22-26 in arXiv:1508.07253
  */
__device__
double delta0_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f22 = d->f22;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((d2*f15*f22*f3 - 2*d2*f14*f23*f3 + d2*f13*f24*f3 - d2*f15*f2*f32 + d2*f14*f22*f32
  - d1*f13*f23*f32 + d2*f13*f23*f32 + d1*f12*f24*f32 - d2*f12*f24*f32 + d2*f14*f2*f33
  + 2*d1*f13*f22*f33 - 2*d2*f13*f22*f33 - d1*f12*f23*f33 + d2*f12*f23*f33 - d1*f1*f24*f33
  - d1*f13*f2*f34 - d1*f12*f22*f34 + 2*d1*f1*f23*f34 + d1*f12*f2*f35 - d1*f1*f22*f35
  + 4*f12*f23*f32*v1 - 3*f1*f24*f32*v1 - 8*f12*f22*f33*v1 + 4*f1*f23*f33*v1 + f24*f33*v1
  + 4*f12*f2*f34*v1 + f1*f22*f34*v1 - 2*f23*f34*v1 - 2*f1*f2*f35*v1 + f22*f35*v1 - f15*f32*v2
  + 3*f14*f33*v2 - 3*f13*f34*v2 + f12*f35*v2 - f15*f22*v3 + 2*f14*f23*v3 - f13*f24*v3
  + 2*f15*f2*f3*v3 - f14*f22*f3*v3 - 4*f13*f23*f3*v3 + 3*f12*f24*f3*v3 - 4*f14*f2*f32*v3
  + 8*f13*f22*f32*v3 - 4*f12*f23*f32*v3) / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(f3-f2)));
}

__device__
double delta1_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f22 = d->f22;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((-(d2*f15*f22) + 2*d2*f14*f23 - d2*f13*f24 - d2*f14*f22*f3 + 2*d1*f13*f23*f3
  + 2*d2*f13*f23*f3 - 2*d1*f12*f24*f3 - d2*f12*f24*f3 + d2*f15*f32 - 3*d1*f13*f22*f32
  - d2*f13*f22*f32 + 2*d1*f12*f23*f32 - 2*d2*f12*f23*f32 + d1*f1*f24*f32 + 2*d2*f1*f24*f32
  - d2*f14*f33 + d1*f12*f22*f33 + 3*d2*f12*f22*f33 - 2*d1*f1*f23*f33 - 2*d2*f1*f23*f33
  + d1*f24*f33 + d1*f13*f34 + d1*f1*f22*f34 - 2*d1*f23*f34 - d1*f12*f35 + d1*f22*f35
  - 8*f12*f23*f3*v1 + 6*f1*f24*f3*v1 + 12*f12*f22*f32*v1 - 8*f1*f23*f32*v1 - 4*f12*f34*v1
  + 2*f1*f35*v1 + 2*f15*f3*v2 - 4*f14*f32*v2 + 4*f12*f34*v2 - 2*f1*f35*v2 - 2*f15*f3*v3
  + 8*f12*f23*f3*v3 - 6*f1*f24*f3*v3 + 4*f14*f32*v3 - 12*f12*f22*f32*v3 + 8*f1*f23*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

__device__
double delta2_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f15 = d->f15;
  double f23 = d->f23;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;
  double f35 = d->f35;

  return -((d2*f15*f2 - d1*f13*f23 - 3*d2*f13*f23 + d1*f12*f24 + 2*d2*f12*f24 - d2*f15*f3
  + d2*f14*f2*f3 - d1*f12*f23*f3 + d2*f12*f23*f3 + d1*f1*f24*f3 - d2*f1*f24*f3 - d2*f14*f32
  + 3*d1*f13*f2*f32 + d2*f13*f2*f32 - d1*f1*f23*f32 + d2*f1*f23*f32 - 2*d1*f24*f32 - d2*f24*f32
  - 2*d1*f13*f33 + 2*d2*f13*f33 - d1*f12*f2*f33 - 3*d2*f12*f2*f33 + 3*d1*f23*f33 + d2*f23*f33
  + d1*f12*f34 - d1*f1*f2*f34 + d1*f1*f35 - d1*f2*f35 + 4*f12*f23*v1 - 3*f1*f24*v1 + 4*f1*f23*f3*v1
  - 3*f24*f3*v1 - 12*f12*f2*f32*v1 + 4*f23*f32*v1 + 8*f12*f33*v1 - f1*f34*v1 - f35*v1 - f15*v2
  - f14*f3*v2 + 8*f13*f32*v2 - 8*f12*f33*v2 + f1*f34*v2 + f35*v2 + f15*v3 - 4*f12*f23*v3 + 3*f1*f24*v3
  + f14*f3*v3 - 4*f1*f23*f3*v3 + 3*f24*f3*v3 - 8*f13*f32*v3 + 12*f12*f2*f32*v3 - 4*f23*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

__device__
double delta3_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f14 = d->f14;
  double f22 = d->f22;
  double f24 = d->f24;
  double f32 = d->f32;
  double f33 = d->f33;
  double f34 = d->f34;

  return -((-2*d2*f14*f2 + d1*f13*f22 + 3*d2*f13*f22 - d1*f1*f24 - d2*f1*f24 + 2*d2*f14*f3
  - 2*d1*f13*f2*f3 - 2*d2*f13*f2*f3 + d1*f12*f22*f3 - d2*f12*f22*f3 + d1*f24*f3 + d2*f24*f3
  + d1*f13*f32 - d2*f13*f32 - 2*d1*f12*f2*f32 + 2*d2*f12*f2*f32 + d1*f1*f22*f32 - d2*f1*f22*f32
  + d1*f12*f33 - d2*f12*f33 + 2*d1*f1*f2*f33 + 2*d2*f1*f2*f33 - 3*d1*f22*f33 - d2*f22*f33
  - 2*d1*f1*f34 + 2*d1*f2*f34 - 4*f12*f22*v1 + 2*f24*v1 + 8*f12*f2*f3*v1 - 4*f1*f22*f3*v1
  - 4*f12*f32*v1 + 8*f1*f2*f32*v1 - 4*f22*f32*v1 - 4*f1*f33*v1 + 2*f34*v1 + 2*f14*v2
  - 4*f13*f3*v2 + 4*f1*f33*v2 - 2*f34*v2 - 2*f14*v3 + 4*f12*f22*v3 - 2*f24*v3 + 4*f13*f3*v3
  - 8*f12*f2*f3*v3 + 4*f1*f22*f3*v3 + 4*f12*f32*v3 - 8*f1*f2*f32*v3 + 4*f22*f32*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

__device__
double delta4_fun(IMRPhenomDAmplitudeCoefficients* p, DeltaUtility* d) {
  double f1 = p->f1;
  double f2 = p->f2;
  double f3 = p->f3;
  double v1 = p->v1;
  double v2 = p->v2;
  double v3 = p->v3;
  double d1 = p->d1;
  double d2 = p->d2;

  double f12 = d->f12;
  double f13 = d->f13;
  double f22 = d->f22;
  double f23 = d->f23;
  double f32 = d->f32;
  double f33 = d->f33;

  return -((d2*f13*f2 - d1*f12*f22 - 2*d2*f12*f22 + d1*f1*f23 + d2*f1*f23 - d2*f13*f3 + 2*d1*f12*f2*f3
  + d2*f12*f2*f3 - d1*f1*f22*f3 + d2*f1*f22*f3 - d1*f23*f3 - d2*f23*f3 - d1*f12*f32 + d2*f12*f32
  - d1*f1*f2*f32 - 2*d2*f1*f2*f32 + 2*d1*f22*f32 + d2*f22*f32 + d1*f1*f33 - d1*f2*f33 + 3*f1*f22*v1
  - 2*f23*v1 - 6*f1*f2*f3*v1 + 3*f22*f3*v1 + 3*f1*f32*v1 - f33*v1 - f13*v2 + 3*f12*f3*v2 - 3*f1*f32*v2
  + f33*v2 + f13*v3 - 3*f1*f22*v3 + 2*f23*v3 - 3*f12*f3*v3 + 6*f1*f2*f3*v3 - 3*f22*f3*v3)
  / (pow_2_of(f1 - f2)*pow_3_of(f1 - f3)*pow_2_of(-f2 + f3)));
}

/**
 * Calculates delta_i's
 * Method described in arXiv:1508.07253 section 'Region IIa - intermediate'
 */
__device__
void ComputeDeltasFromCollocation(IMRPhenomDAmplitudeCoefficients* p) {
  // Three evenly spaced collocation points in the interval [f1,f3].
  double f1 = AMP_fJoin_INS;
  double f3 = p->fmaxCalc;
  double dfx = (f3 - f1)/2.0;
  double f2 = f1 + dfx;

  UsefulPowers powers_of_f1;
  int status = init_useful_powers(&powers_of_f1, f1);

  AmpInsPrefactors prefactors;
  status = init_amp_ins_prefactors(&prefactors, p);

  // v1 is inspiral model evaluated at f1
  // d1 is derivative of inspiral model evaluated at f1
  double v1 = AmpInsAnsatz(f1, &powers_of_f1, &prefactors);
  double d1 = DAmpInsAnsatz(f1, p);

  // v3 is merger-ringdown model evaluated at f3
  // d2 is derivative of merger-ringdown model evaluated at f3
  double v3 = AmpMRDAnsatz(f3, p);
  double d2 = DAmpMRDAnsatz(f3, p);

  // v2 is the value of the amplitude evaluated at f2
  // they come from the fit of the collocation points in the intermediate region
  double v2 = AmpIntColFitCoeff(p->eta, p->chi);

  p->f1 = f1;
  p->f2 = f2;
  p->f3 = f3;
  p->v1 = v1;
  p->v2 = v2;
  p->v3 = v3;
  p->d1 = d1;
  p->d2 = d2;

  // Now compute the delta_i's from the collocation coefficients
  // Precompute common quantities here and pass along to delta functions.
  DeltaUtility d;
  d.f12 = f1*f1;
  d.f13 = f1*d.f12;
  d.f14 = f1*d.f13;
  d.f15 = f1*d.f14;
  d.f22 = f2*f2;
  d.f23 = f2*d.f22;
  d.f24 = f2*d.f23;
  d.f32 = f3*f3;
  d.f33 = f3*d.f32;
  d.f34 = f3*d.f33;
  d.f35 = f3*d.f34;
  p->delta0 = delta0_fun(p, &d);
  p->delta1 = delta1_fun(p, &d);
  p->delta2 = delta2_fun(p, &d);
  p->delta3 = delta3_fun(p, &d);
  p->delta4 = delta4_fun(p, &d);
}


/**
 * A struct containing all the parameters that need to be calculated
 * to compute the phenomenological amplitude
 */
__device__
void ComputeIMRPhenomDAmplitudeCoefficients(IMRPhenomDAmplitudeCoefficients* p, double eta, double chi1, double chi2, double fring, double fdamp) {
  p->eta = eta;
  p->chi1 = chi1;
  p->chi2 = chi2;

  p->q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  p->chi = chiPN(eta, chi1, chi2);

  p->fRD = fring;
  p->fDM = fdamp;

  // Compute gamma_i's, rho_i's first then delta_i's
  p->gamma1 = gamma1_fun(eta, p->chi);
  p->gamma2 = gamma2_fun(eta, p->chi);
  p->gamma3 = gamma3_fun(eta, p->chi);

  p->fmaxCalc = fmaxCalc(p);

  p->rho1 = rho1_fun(eta, p->chi);
  p->rho2 = rho2_fun(eta, p->chi);
  p->rho3 = rho3_fun(eta, p->chi);

  // compute delta_i's
  ComputeDeltasFromCollocation(p);
}


/********************************* Phase functions *********************************/

////////////////////////////// Phase: Ringdown functions ///////////////////////////

// alpha_i i=1,2,3,4,5 are the phenomenological intermediate coefficients depending on eta and chiPN
// PhiRingdownAnsatz is the ringdown phasing in terms of the alpha_i coefficients

/**
 * alpha 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double alpha1Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 43.31514709695348 + 638.6332679188081*eta
    + (-32.85768747216059 + 2415.8938269370315*eta - 5766.875169379177*eta2)*xi
    + (-61.85459307173841 + 2953.967762459948*eta - 8986.29057591497*eta2)*xi2
    + (-21.571435779762044 + 981.2158224673428*eta - 3239.5664895930286*eta2)*xi3;
}

/**
 * alpha 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double alpha2Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -0.07020209449091723 - 0.16269798450687084*eta
  + (-0.1872514685185499 + 1.138313650449945*eta - 2.8334196304430046*eta2)*xi
  + (-0.17137955686840617 + 1.7197549338119527*eta - 4.539717148261272*eta2)*xi2
  + (-0.049983437357548705 + 0.6062072055948309*eta - 1.682769616644546*eta2)*xi3;
}

/**
 * alpha 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double alpha3Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 9.5988072383479 - 397.05438595557433*eta
  + (16.202126189517813 - 1574.8286986717037*eta + 3600.3410843831093*eta2)*xi
  + (27.092429659075467 - 1786.482357315139*eta + 5152.919378666511*eta2)*xi2
  + (11.175710130033895 - 577.7999423177481*eta + 1808.730762932043*eta2)*xi3;
}

/**
 * alpha 4 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double alpha4Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -0.02989487384493607 + 1.4022106448583738*eta
  + (-0.07356049468633846 + 0.8337006542278661*eta + 0.2240008282397391*eta2)*xi
  + (-0.055202870001177226 + 0.5667186343606578*eta + 0.7186931973380503*eta2)*xi2
  + (-0.015507437354325743 + 0.15750322779277187*eta + 0.21076815715176228*eta2)*xi3;
}

/**
 * alpha 5 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double alpha5Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 0.9974408278363099 - 0.007884449714907203*eta
  + (-0.059046901195591035 + 1.3958712396764088*eta - 4.516631601676276*eta2)*xi
  + (-0.05585343136869692 + 1.7516580039343603*eta - 5.990208965347804*eta2)*xi2
  + (-0.017945336522161195 + 0.5965097794825992*eta - 2.0608879367971804*eta2)*xi3;
}

/**
 * Ansatz for the merger-ringdown phase Equation 14 arXiv:1508.07253
 * Rholm was added when IMRPhenomHM (high mode) was added.
 * Rholm = fRD22/fRDlm. For PhenomD (only (l,m)=(2,2)) this is just equal
 * to 1. and PhenomD is recovered.
 * Taulm = fDMlm/fDM22. Ratio of ringdown damping times.
 * Again, when Taulm = 1.0 then PhenomD is recovered.
 */
__device__
double PhiMRDAnsatzInt(double f, IMRPhenomDPhaseCoefficients *p, double Rholm, double Taulm)
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

/**
 * First frequency derivative of PhiMRDAnsatzInt
 * Rholm was added when IMRPhenomHM (high mode) was added.
 * Rholm = fRD22/fRDlm. For PhenomD (only (l,m)=(2,2)) this is just equal
 * to 1. and PhenomD is recovered.
 * Taulm = fDMlm/fDM22. Ratio of ringdown damping times.
 * Again, when Taulm = 1.0 then PhenomD is recovered.
 */
 __device__
double DPhiMRD(double f, IMRPhenomDPhaseCoefficients *p, double Rholm, double Taulm) {
  return ( p->alpha1 + p->alpha2/pow_2_of(f) + p->alpha3/pow(f,0.25)+ p->alpha4/(p->fDM * Taulm * (1 + pow_2_of(f - p->alpha5 * p->fRD)/(pow_2_of(p->fDM * Taulm * Rholm)))) ) * p->etaInv;
}


///////////////////////////// Phase: Intermediate functions /////////////////////////////

// beta_i i=1,2,3 are the phenomenological intermediate coefficients depending on eta and chiPN
// PhiIntAnsatz is the intermediate phasing in terms of the beta_i coefficients


// \[Beta]1Fit = PhiIntFitCoeff\[Chi]PNFunc[\[Eta], \[Chi]PN][[1]]

/**
 * beta 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double beta1Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 97.89747327985583 - 42.659730877489224*eta
  + (153.48421037904913 - 1417.0620760768954*eta + 2752.8614143665027*eta2)*xi
  + (138.7406469558649 - 1433.6585075135881*eta + 2857.7418952430758*eta2)*xi2
  + (41.025109467376126 - 423.680737974639*eta + 850.3594335657173*eta2)*xi3;
}

/**
 * beta 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double beta2Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -3.282701958759534 - 9.051384468245866*eta
  + (-12.415449742258042 + 55.4716447709787*eta - 106.05109938966335*eta2)*xi
  + (-11.953044553690658 + 76.80704618365418*eta - 155.33172948098394*eta2)*xi2
  + (-3.4129261592393263 + 25.572377569952536*eta - 54.408036707740465*eta2)*xi3;
}

/**
 * beta 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double beta3Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -0.000025156429818799565 + 0.000019750256942201327*eta
  + (-0.000018370671469295915 + 0.000021886317041311973*eta + 0.00008250240316860033*eta2)*xi
  + (7.157371250566708e-6 - 0.000055780000112270685*eta + 0.00019142082884072178*eta2)*xi2
  + (5.447166261464217e-6 - 0.00003220610095021982*eta + 0.00007974016714984341*eta2)*xi3;
}

/**
 * ansatz for the intermediate phase defined by Equation 16 arXiv:1508.07253
 */
__device__
double PhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
  // 1./eta in paper omitted and put in when need in the functions:
  // ComputeIMRPhenDPhaseConnectionCoefficients
  // IMRPhenDPhase
  return  p->beta1*Mf - p->beta3/(3.*pow_3_of(Mf)) + p->beta2*log(Mf);
}

/**
 * First frequency derivative of PhiIntAnsatz
 * (this time with 1./eta explicitly factored in)
 */
__device__
double DPhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
  return (p->beta1 + p->beta3/pow_4_of(Mf) + p->beta2/Mf) / p->eta;
}

/**
 * temporary instance of DPhiIntAnsatz used when computing
 * coefficients to make the phase C(1) continuous between regions.
 */
 __device__
double DPhiIntTemp(double ff, IMRPhenomDPhaseCoefficients *p) {
  double eta = p->eta;
  double beta1 = p->beta1;
  double beta2 = p->beta2;
  double beta3 = p->beta3;
  double C2Int = p->C2Int;

  return C2Int + (beta1 + beta3/pow_4_of(ff) + beta2/ff)/eta;
}


///////////////////////////// Phase: Inspiral functions /////////////////////////////

// sigma_i i=1,2,3,4 are the phenomenological inspiral coefficients depending on eta and chiPN
// PhiInsAnsatzInt is a souped up TF2 phasing which depends on the sigma_i coefficients

/**
 * sigma 1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double sigma1Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 2096.551999295543 + 1463.7493168261553*eta
  + (1312.5493286098522 + 18307.330017082117*eta - 43534.1440746107*eta2)*xi
  + (-833.2889543511114 + 32047.31997183187*eta - 108609.45037520859*eta2)*xi2
  + (452.25136398112204 + 8353.439546391714*eta - 44531.3250037322*eta2)*xi3;
}

/**
 * sigma 2 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double sigma2Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -10114.056472621156 - 44631.01109458185*eta
  + (-6541.308761668722 - 266959.23419307504*eta + 686328.3229317984*eta2)*xi
  + (3405.6372187679685 - 437507.7208209015*eta + 1.6318171307344697e6*eta2)*xi2
  + (-7462.648563007646 - 114585.25177153319*eta + 674402.4689098676*eta2)*xi3;
}

/**
 * sigma 3 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double sigma3Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return 22933.658273436497 + 230960.00814979506*eta
  + (14961.083974183695 + 1.1940181342318142e6*eta - 3.1042239693052764e6*eta2)*xi
  + (-3038.166617199259 + 1.8720322849093592e6*eta - 7.309145012085539e6*eta2)*xi2
  + (42738.22871475411 + 467502.018616601*eta - 3.064853498512499e6*eta2)*xi3;
}

/**
 * sigma 4 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
__device__
double sigma4Fit(double eta, double chi) {
  double xi = -1 + chi;
  double xi2 = xi*xi;
  double xi3 = xi2*xi;
  double eta2 = eta*eta;

  return -14621.71522218357 - 377812.8579387104*eta
  + (-9608.682631509726 - 1.7108925257214056e6*eta + 4.332924601416521e6*eta2)*xi
  + (-22366.683262266528 - 2.5019716386377467e6*eta + 1.0274495902259542e7*eta2)*xi2
  + (-85360.30079034246 - 570025.3441737515*eta + 4.396844346849777e6*eta2)*xi3;
}

/**
 * Ansatz for the inspiral phase.
 * We call the LAL TF2 coefficients here.
 * The exact values of the coefficients used are given
 * as comments in the top of this file
 * Defined by Equation 27 and 28 arXiv:1508.07253
 */

__device__
double PhiInsAnsatzInt(double Mf, UsefulPowers *powers_of_Mf, PhiInsPrefactors *prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn)
{
	//assert(0 != pn);

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
int init_phi_ins_prefactors(PhiInsPrefactors * prefactors, IMRPhenomDPhaseCoefficients* p, PNPhasingSeries *pn)
{

	double sigma1 = p->sigma1;
	double sigma2 = p->sigma2;
	double sigma3 = p->sigma3;
	double sigma4 = p->sigma4;
	double Pi = PI;

    UsefulPowers powers_of_pi;
    init_useful_powers(&powers_of_pi, PI);

  // PN phasing series
	prefactors->initial_phasing = pn->v5 - PI_4;
	prefactors->two_thirds = pn->v7 * powers_of_pi.two_thirds;
	prefactors->third = pn->v6 * powers_of_pi.third;
	prefactors->third_with_logv = pn->vlogv6 * powers_of_pi.third;
	prefactors->logv = pn->vlogv5;
	prefactors->minus_third = pn->v4 / powers_of_pi.third;
	prefactors->minus_two_thirds = pn->v3 / powers_of_pi.two_thirds;
	prefactors->minus_one = pn->v2 / Pi;
	prefactors->minus_five_thirds = pn->v0 / powers_of_pi.five_thirds; // * v^0

  // higher order terms that were calibrated for PhenomD
	prefactors->one = sigma1;
	prefactors->four_thirds = sigma2 * 3.0/4.0;
	prefactors->five_thirds = sigma3 * 3.0/5.0;
	prefactors->two = sigma4 / 2.0;

	return 1;
}

/**
 * First frequency derivative of PhiInsAnsatzInt
 */
__device__
double DPhiInsAnsatzInt(double Mf, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn) {
  double sigma1 = p->sigma1;
  double sigma2 = p->sigma2;
  double sigma3 = p->sigma3;
  double sigma4 = p->sigma4;
  double Pi = PI;

  // Assemble PN phasing series
  const double v = cbrt(Pi*Mf);
  const double logv = log(v);
  const double v2 = v * v;
  const double v3 = v * v2;
  const double v4 = v * v3;
  const double v5 = v * v4;
  const double v6 = v * v5;
  const double v7 = v * v6;
  const double v8 = v * v7;

  // Apply the correct prefactors to LAL phase coefficients to get the
  // phase derivative dphi / dMf = dphi/dv * dv/dMf
  double Dphasing = 0.0;
  Dphasing += +2.0 * pn->v7 * v7;
  Dphasing += (pn->v6 + pn->vlogv6 * (1.0 + logv)) * v6;
  Dphasing += pn->vlogv5 * v5;
  Dphasing += -1.0 * pn->v4 * v4;
  Dphasing += -2.0 * pn->v3 * v3;
  Dphasing += -3.0 * pn->v2 * v2;
  Dphasing += -4.0 * pn->v1 * v;
  Dphasing += -5.0 * pn->v0;
  Dphasing /= v8 * 3.0/Pi;

  // Now add higher order terms that were calibrated for PhenomD
  Dphasing += (
          sigma1
        + sigma2 * v / pow(PI, 1./3.) //powers_of_pi.third
        + sigma3 * v2 / pow(PI, 2./3.) //powers_of_pi.two_thirds
        + (sigma4/Pi) * v3
        ) / p->eta;

  return Dphasing;
}



/**
 * A struct containing all the parameters that need to be calculated
 * to compute the phenomenological phase
 */
 __device__
 void ComputeIMRPhenomDPhaseCoefficients(IMRPhenomDPhaseCoefficients* p, double eta, double chi1, double chi2, double fring, double fdamp) {

  // Convention m1 >= m2
  p->eta = eta;
  p->etaInv = 1.0/eta;
  p->chi1 = chi1;
  p->chi2 = chi2;

  p->q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta);
  p->chi = chiPN(eta, chi1, chi2);

  p->sigma1 = sigma1Fit(eta, p->chi);
  p->sigma2 = sigma2Fit(eta, p->chi);
  p->sigma3 = sigma3Fit(eta, p->chi);
  p->sigma4 = sigma4Fit(eta, p->chi);

  p->beta1 = beta1Fit(eta, p->chi);
  p->beta2 = beta2Fit(eta, p->chi);
  p->beta3 = beta3Fit(eta, p->chi);

  p->alpha1 = alpha1Fit(eta, p->chi);
  p->alpha2 = alpha2Fit(eta, p->chi);
  p->alpha3 = alpha3Fit(eta, p->chi);
  p->alpha4 = alpha4Fit(eta, p->chi);
  p->alpha5 = alpha5Fit(eta, p->chi);

  p->fRD = fring;
  p->fDM = fdamp;

}

/**
 * This function aligns the three phase parts (inspiral, intermediate and merger-rindown)
 * such that they are c^1 continuous at the transition frequencies
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 * Rholm was added when IMRPhenomHM (high mode) was added.
 * Rholm = fRD22/fRDlm. For PhenomD (only (l,m)=(2,2)) this is just equal
 * to 1. and PhenomD is recovered.
 * Taulm = fDMlm/fDM22. Ratio of ringdown damping times.
 * Again, when Taulm = 1.0 then PhenomD is recovered.
 */
__device__
void ComputeIMRPhenDPhaseConnectionCoefficients(IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, PhiInsPrefactors *prefactors, double Rholm, double Taulm)
{
  double etaInv = p->etaInv;

  // Transition frequencies
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  p->fInsJoin=PHI_fJoin_INS;
  p->fMRDJoin=0.5*p->fRD;

  // Compute C1Int and C2Int coeffs
  // Equations to solve for to get C(1) continuous join
  // PhiIns (f)  =   PhiInt (f) + C1Int + C2Int f
  // Joining at fInsJoin
  // PhiIns (fInsJoin)  =   PhiInt (fInsJoin) + C1Int + C2Int fInsJoin
  // PhiIns'(fInsJoin)  =   PhiInt'(fInsJoin) + C2Int
  double DPhiIns = DPhiInsAnsatzInt(PHI_fJoin_INS, p, pn);
  double DPhiInt = DPhiIntAnsatz(PHI_fJoin_INS, p);
  p->C2Int = DPhiIns - DPhiInt;

  UsefulPowers powers_of_fInsJoin;
  init_useful_powers(&powers_of_fInsJoin, PHI_fJoin_INS);
  p->C1Int = PhiInsAnsatzInt(PHI_fJoin_INS, &powers_of_fInsJoin, prefactors, p, pn)
    - etaInv * PhiIntAnsatz(PHI_fJoin_INS, p) - p->C2Int * PHI_fJoin_INS;

  // Compute C1MRD and C2MRD coeffs
  // Equations to solve for to get C(1) continuous join
  // PhiInsInt (f)  =   PhiMRD (f) + C1MRD + C2MRD f
  // Joining at fMRDJoin
  // Where \[Phi]InsInt(f) is the \[Phi]Ins+\[Phi]Int joined function
  // PhiInsInt (fMRDJoin)  =   PhiMRD (fMRDJoin) + C1MRD + C2MRD fMRDJoin
  // PhiInsInt'(fMRDJoin)  =   PhiMRD'(fMRDJoin) + C2MRD
  // temporary Intermediate Phase function to Join up the Merger-Ringdown
  double PhiIntTempVal = etaInv * PhiIntAnsatz(p->fMRDJoin, p) + p->C1Int + p->C2Int*p->fMRDJoin;
  double DPhiIntTempVal = DPhiIntTemp(p->fMRDJoin, p);
  double DPhiMRDVal = DPhiMRD(p->fMRDJoin, p, Rholm, Taulm);
  p->C2MRD = DPhiIntTempVal - DPhiMRDVal;
  p->C1MRD = PhiIntTempVal - etaInv * PhiMRDAnsatzInt(p->fMRDJoin, p, Rholm, Taulm) - p->C2MRD*p->fMRDJoin;
}


/**
 * Step function in boolean version
 */
__device__
bool StepFunc_boolean(const double t, const double t1) {
	return (t >= t1);
}



/**
 * This function computes the IMR phase given phenom coefficients.
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 * Rholm was added when IMRPhenomHM (high mode) was added.
 * Rholm = fRD22/fRDlm. For PhenomD (only (l,m)=(2,2)) this is just equal
 * to 1. and PhenomD is recovered.
 * Taulm = fDMlm/fDM22. Ratio of ringdown damping times.
 * Again, when Taulm = 1.0 then PhenomD is recovered.
 */
 __device__
double IMRPhenDPhase(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, UsefulPowers *powers_of_f, PhiInsPrefactors *prefactors, double Rholm, double Taulm)
{
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermendiate and merger-ringdown phase parts

  // split the calculation to just 1 of 3 possible mutually exclusive ranges
  if (!StepFunc_boolean(f, p->fInsJoin))	// Inspiral range
  {
	  double PhiIns = PhiInsAnsatzInt(f, powers_of_f, prefactors, p, pn);
	  return PhiIns;
  }

  if (StepFunc_boolean(f, p->fMRDJoin))	// MRD range
  {
	  double PhiMRD = p->etaInv * PhiMRDAnsatzInt(f, p, Rholm, Taulm) + p->C1MRD + p->C2MRD * f;
	  return PhiMRD;
  }

  //	Intermediate range
  double PhiInt = p->etaInv * PhiIntAnsatz(f, p) + p->C1Int + p->C2Int * f;
  return PhiInt;
}

/**
 * Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
 * (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
 * was not available when PhenomD was tuned.
 */
__device__
double Subtract3PNSS(double m1, double m2, double M, double eta, double chi1, double chi2){
  double m1M = m1 / M;
  double m2M = m2 / M;
  double pn_ss3 =  (326.75L/1.12L + 557.5L/1.8L*eta)*eta*chi1*chi2;
  pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m1M-120.L*m1M*m1M) + (-4108.25L/6.72L-108.5L/1.2L*m1M+125.5L/3.6L*m1M*m1M)) *m1M*m1M * chi1*chi1;
  pn_ss3 += ((4703.5L/8.4L+2935.L/6.L*m2M-120.L*m2M*m2M) + (-4108.25L/6.72L-108.5L/1.2L*m2M+125.5L/3.6L*m2M*m2M)) *m2M*m2M * chi2*chi2;
  return pn_ss3;
}


/**
 * Convert from geometric frequency to frequency in Hz
 */
 __device__
double PhenomUtilsMftoHz(
    double Mf,       /**< Geometric frequency */
    double Mtot_Msun /**< Total mass in solar masses */
)
{
    return Mf / (MTSUN_SI * Mtot_Msun);
}

/**
 * Convert from frequency in Hz to geometric frequency
 */
 __device__
double PhenomUtilsHztoMf(
    double fHz,      /**< Frequency in Hz */
    double Mtot_Msun /**< Total mass in solar masses */
)
{
    return fHz * (MTSUN_SI * Mtot_Msun);
}


/**
 * compute the frequency domain amplitude pre-factor
 */
__device__
double PhenomUtilsFDamp0(
    double Mtot_Msun, /**< total mass in solar masses */
    double distance   /**< distance (m) */
)
{
    return Mtot_Msun * MRSUN_SI * Mtot_Msun * MTSUN_SI / distance;
}


/**
 * Given m1 with aligned-spin chi1z and m2 with aligned-spin chi2z.
 * Enforce that m1 >= m2 and swap spins accordingly.
 * Enforce that the primary object (heavier) is indexed by 1.
 * To be used with aligned-spin waveform models.
 * TODO: There is another function for precessing waveform models
 */
__device__
int PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
    double *m1,    /**< [out] mass of body 1 */
    double *m2,    /**< [out] mass of body 2 */
    double *chi1z, /**< [out] aligned-spin component of body 1 */
    double *chi2z  /**< [out] aligned-spin component of body 2 */
)
{
    double chi1z_tmp, chi2z_tmp, m1_tmp, m2_tmp;
    if (*m1 > *m2)
    {
        chi1z_tmp = *chi1z;
        chi2z_tmp = *chi2z;
        m1_tmp = *m1;
        m2_tmp = *m2;
    }
    else
    { /* swap spins and masses */
        chi1z_tmp = *chi2z;
        chi2z_tmp = *chi1z;
        m1_tmp = *m2;
        m2_tmp = *m1;
    }
    *m1 = m1_tmp;
    *m2 = m2_tmp;
    *chi1z = chi1z_tmp;
    *chi2z = chi2z_tmp;

        //ERROR(PD_EDOM, "ERROR in EnforcePrimaryIsm1. When trying\
 //to enfore that m1 should be the larger mass.\
 //After trying to enforce this m1 = %f and m2 = %f\n");

    return 1;
}



// Call ComputeIMRPhenomDAmplitudeCoefficients() first!
/**
 * This function computes the IMR amplitude given phenom coefficients.
 * Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 */
__device__
double IMRPhenDAmplitude(double f, IMRPhenomDAmplitudeCoefficients *p, UsefulPowers *powers_of_f, AmpInsPrefactors * prefactors) {
  // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
  // The inspiral, intermediate and merger-ringdown amplitude parts

  // Transition frequencies
  p->fInsJoin = AMP_fJoin_INS;
  p->fMRDJoin = p->fmaxCalc;

  double f_seven_sixths = f * powers_of_f->sixth;
  double AmpPreFac = prefactors->amp0 / f_seven_sixths;

  // split the calculation to just 1 of 3 possible mutually exclusive ranges

  if (!StepFunc_boolean(f, p->fInsJoin))	// Inspiral range
  {
	  double AmpIns = AmpPreFac * AmpInsAnsatz(f, powers_of_f, prefactors);
	  return AmpIns;
  }

  if (StepFunc_boolean(f, p->fMRDJoin))	// MRD range
  {
	  double AmpMRD = AmpPreFac * AmpMRDAnsatz(f, p);
	  return AmpMRD;
  }

  //	Intermediate range
  double AmpInt = AmpPreFac * AmpIntAnsatz(f, p);
  return AmpInt;
}


/**
 * computes the time shift as the approximate time of the peak of the 22 mode.
 */
__device__
double IMRPhenomDComputet0(
    double eta,           /**< symmetric mass-ratio */
    double chi1z,         /**< dimensionless aligned-spin of primary */
    double chi2z,         /**< dimensionless aligned-spin of secondary */
    double finspin,       /**< final spin */
    IMRPhenomDPhaseCoefficients *pPhi,
    IMRPhenomDAmplitudeCoefficients *pAmp
)
{

  //time shift so that peak amplitude is approximately at t=0
  //For details see https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/timedomain
  //NOTE: All modes will have the same time offset. So we use the 22 mode.
  //If we just use the 22 mode then we pass 1.0, 1.0 into DPhiMRD.
  double t0 = DPhiMRD(pAmp->fmaxCalc, pPhi, 1.0, 1.0);

  return t0;
}



/**
 * Function to return the phenomD phase using the
 * IMRPhenomDSetupAmpAndPhaseCoefficients struct
 */

__device__
 double IMRPhenomDPhase_OneFrequency(
    double Mf,
    PhenDAmpAndPhasePreComp pD,
    double Rholm,
    double Taulm)
{

  UsefulPowers powers_of_f;
  int status = init_useful_powers(&powers_of_f, Mf);
  //assert(1 == status) ; //, status, "Failed to initiate init_useful_powers");
  double phase = IMRPhenDPhase(Mf, &(pD.pPhi), &(pD.pn), &powers_of_f,
                              &(pD.phi_prefactors), Rholm, Taulm);
  return phase;
}


/**
* Function to compute the amplitude and phase coefficients for PhenomD
* Used to optimise the calls to IMRPhenDPhase and IMRPhenDAmplitude
*/
__device__
int IMRPhenomDSetupAmpAndPhaseCoefficients(
   PhenDAmpAndPhasePreComp *pDPreComp,
   double m1,
   double m2,
   double chi1z,
   double chi2z,
   const double Rholm,
   const double Taulm,
   double fring,
   double fdamp)
{

 /* It's difficult to see in the code but you need to setup the
    * powers_of_pi.
    */
 int retcode = 0;
 UsefulPowers powers_of_pi;
 retcode = init_useful_powers(&powers_of_pi, PI);

 PhenomInternal_AlignedSpinEnforcePrimaryIsm1(&m1, &m2, &chi1z, &chi2z);
 const double Mtot = m1 + m2;
 const double eta = m1 * m2 / (Mtot * Mtot);

 // Left in for historical record

 //if (finspin < MIN_FINAL_SPIN)
   //printf("Final spin (Mf=%g) and ISCO frequency of this system are small, \
    //                       the model might misbehave here.",
    //                  finspin);

 //start phase

ComputeIMRPhenomDPhaseCoefficients(&pDPreComp->pPhi, eta, chi1z, chi2z, fring, fdamp);

 TaylorF2AlignedPhasing(&pDPreComp->pn, m1, m2, chi1z, chi2z);

 // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
 // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
 // was not available when PhenomD was tuned.
 pDPreComp->pn.v6 -= (Subtract3PNSS(m1, m2, Mtot, eta, chi1z, chi2z) * pDPreComp->pn.v0);

 retcode = 0;
 retcode = init_phi_ins_prefactors(&pDPreComp->phi_prefactors, &pDPreComp->pPhi, &pDPreComp->pn);

 // Compute coefficients to make phase C^1 continuous (phase and first derivative)
 ComputeIMRPhenDPhaseConnectionCoefficients(&pDPreComp->pPhi, &pDPreComp->pn, &pDPreComp->phi_prefactors, Rholm, Taulm);
 //end phase

 //start amp
 ComputeIMRPhenomDAmplitudeCoefficients(&pDPreComp->pAmp, eta, chi1z, chi2z, fring, fdamp);

 retcode = 0;
 retcode = init_amp_ins_prefactors(&pDPreComp->amp_prefactors, &pDPreComp->pAmp);
//end amp

 //output
 return 1;
}




/**
 * returns the real and imag parts of the complex ringdown frequency
 * for the (l,m) mode.
 */
__device__
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


//////////////////////// Final spin, final mass, fring, fdamp ////////////////////////

// Final Spin and Radiated Energy formulas described in 1508.07250

/**
 * Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
 * s defined around Equation 3.6.
 */
__device__
double FinalSpin0815_s(double eta, double s) {
  double eta2 = eta*eta;
  double eta3 = eta2*eta;
  double eta4 = eta3*eta;
  double s2 = s*s;
  double s3 = s2*s;
  double s4 = s3*s;

return 3.4641016151377544*eta - 4.399247300629289*eta2 +
   9.397292189321194*eta3 - 13.180949901606242*eta4 +
   (1 - 0.0850917821418767*eta - 5.837029316602263*eta2)*s +
   (0.1014665242971878*eta - 2.0967746996832157*eta2)*s2 +
   (-1.3546806617824356*eta + 4.108962025369336*eta2)*s3 +
   (-0.8676969352555539*eta + 2.064046835273906*eta2)*s4;
}

/**
 * Wrapper function for FinalSpin0815_s.
 */
__device__
double FinalSpin0815(double eta, double chi1, double chi2) {
  // Convention m1 >= m2
  double Seta = sqrt(1.0 - 4.0*eta);
  double m1 = 0.5 * (1.0 + Seta);
  double m2 = 0.5 * (1.0 - Seta);
  double m1s = m1*m1;
  double m2s = m2*m2;
  // s defined around Equation 3.6 arXiv:1508.07250
  double s = (m1s * chi1 + m2s * chi2);
  return FinalSpin0815_s(eta, s);
}

/**
 * Formula to predict the total radiated energy. Equation 3.7 and 3.8 arXiv:1508.07250
 * Input parameter s defined around Equation 3.7 and 3.8.
 */
__device__
double EradRational0815_s(double eta, double s) {
  double eta2 = eta*eta;
  double eta3 = eta2*eta;
  double eta4 = eta3*eta;

  return ((0.055974469826360077*eta + 0.5809510763115132*eta2 - 0.9606726679372312*eta3 + 3.352411249771192*eta4)*
    (1. + (-0.0030302335878845507 - 2.0066110851351073*eta + 7.7050567802399215*eta2)*s))/(1. + (-0.6714403054720589 - 1.4756929437702908*eta + 7.304676214885011*eta2)*s);
}

/**
 * Wrapper function for EradRational0815_s.
 */
__device__
double EradRational0815(double eta, double chi1, double chi2) {
  // Convention m1 >= m2
  double Seta = sqrt(1.0 - 4.0*eta);
  double m1 = 0.5 * (1.0 + Seta);
  double m2 = 0.5 * (1.0 - Seta);
  double m1s = m1*m1;
  double m2s = m2*m2;
  // arXiv:1508.07250
  double s = (m1s * chi1 + m2s * chi2) / (m1s + m2s);

  return EradRational0815_s(eta, s);
}



/**
 * Helper function used in PhenomHM and PhenomPv3HM
 * Returns the final mass from the fit used in PhenomD
 */
__device__
double IMRPhenomDFinalMass(
    double m1,    /**< mass of primary in solar masses */
    double m2,    /**< mass of secondary in solar masses */
    double chi1z, /**< aligned-spin component on primary */
    double chi2z  /**< aligned-spin component on secondary */
)
{
  int retcode = 0;
  retcode = PhenomInternal_AlignedSpinEnforcePrimaryIsm1(
      &m1,
      &m2,
      &chi1z,
      &chi2z);

  double Mtot = m1 + m2;
  double eta = m1 * m2 / (Mtot * Mtot);

  return (1.0 - EradRational0815(eta, chi1z, chi2z));
}


/**
* Function to return the final spin (spin of the remnant black hole)
* as predicted by the IMRPhenomD model. The final spin is calculated using
* the phenomenological fit described in PhysRevD.93.044006 Eq. 3.6.
* unreviewed
*/
__device__
double IMRPhenomDFinalSpin(
    const double m1_in,                 /**< mass of companion 1 [Msun] */
    const double m2_in,                 /**< mass of companion 2 [Msun] */
    const double chi1_in,               /**< aligned-spin of companion 1 */
    const double chi2_in               /**< aligned-spin of companion 2 */
) {
    // Ensure that m1 > m2 and that chi1 is the spin on m1
    double chi1, chi2, m1, m2;
    if (m1_in>m2_in) {
       chi1 = chi1_in;
       chi2 = chi2_in;
       m1   = m1_in;
       m2   = m2_in;
    } else { // swap spins and masses
       chi1 = chi2_in;
       chi2 = chi1_in;
       m1   = m2_in;
       m2   = m1_in;
    }

    const double M = m1 + m2;
    double eta = m1 * m2 / (M * M);

    double finspin = FinalSpin0815(eta, chi1, chi2);

    /*
    if (finspin < MIN_FINAL_SPIN)
          printf("Final spin and ISCO frequency of this system are small, \
                          the model might misbehave here.");*/

    return finspin;
}


/**
 * Precompute a bunch of PhenomHM related quantities and store them filling in a
 * PhenomHMStorage variable
 */
__device__
static void init_PhenomHM_Storage(
    PhenomHMStorage *p,
    const double m1_SI,
    const double m2_SI,
    const double chi1z,
    const double chi2z,
    const double f_ref,
    const double phiRef
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

    p->finmass = IMRPhenomDFinalMass(p->m1, p->m2, p->chi1z, p->chi2z);
    p->finspin = IMRPhenomDFinalSpin(p->m1, p->m2, p->chi1z, p->chi2z); /* dimensionless final spin */
};

/**
 * domain mapping function - ringdown
 */
__device__
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
        double Rholm = pHM->Rholm;
        ans = Rholm * Mf; /* Used for the Phase */
    }

    return ans;
}

/**
 * mathematica function Ti
 * domain mapping function - inspiral
 */
__device__
double IMRPhenomHMTi(double Mf, const int mm)
{
    return 2.0 * Mf / mm;
}

/**
 * helper function for IMRPhenomHMFreqDomainMap
 */
__device__
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
__device__
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
__device__
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
__device__
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

__device__
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
__device__
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
    cmplx Hlm(0.0, 0.0);
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

    // Compute the final PN Amplitude at Leading Order in fM
    ans = M * M * PI * sqrt(eta * 2.0 / 3) * pow(v, -3.5) * gcmplx::abs(Hlm);

    return ans;
}


/** @} */
/** @} */

__device__
void get_amp(double* amp, double freq_geom, int ell, int mm, PhenomHMStorage* pHM, UsefulPowers powers_of_f, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, double amp0)
{

    double freq_amp, Mf, beta_term1, beta, beta_term2, HMamp_term1, HMamp_term2;

    freq_amp = IMRPhenomHMFreqDomainMap(freq_geom, ell, mm, pHM, AmpFlagTrue);
    Mf = freq_amp; // geometric frequency

    int status_in_for = init_useful_powers(&powers_of_f, Mf);

    double amp_i = IMRPhenDAmplitude(Mf, pAmp, &powers_of_f, &amp_prefactors);

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

    *amp = amp_i*amp0;
}


__device__
void get_phase(double* phase, double freq_geom, int ell, int mm, PhenomHMStorage* pHM, UsefulPowers powers_of_f, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double cshift[], double Rholm, double Taulm, double t0, double phi0)
{
        double Mf_wf, Mfr, tmpphaseC, phase_term1, phase_term2;
      Mf_wf = 0.0;
      double Mf = 0.0;
      Mfr = 0.0;
      tmpphaseC = 0.0;

      double phase_i = cshift[mm];
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

      *phase = (phase_term1 + phase_term2);
}

__device__
double d_dot_product_1d(double* arr1, double* arr2){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}


__device__
cmplx d_vec_H_vec_product(double* arr1, cmplx* H, double* arr2){

    cmplx I(0.0, 1.0);
    cmplx out(0.0, 0.0);
    cmplx trans(0.0, 0.0);
    for (int i=0; i<3; i++){
        trans = cmplx(0.0, 0.0);
        for (int j=0; j<3; j++){
            trans += (H[i*3 + j] * arr2[j]);
        }
        out += arr1[i]*trans;
    }
    return out;
}

__device__
double d_sinc(double x){
    if (x == 0.0) return 1.0;
    else return sin(x)/x;
}


/* # Single-link response
# 'full' does include the orbital-delay term, 'constellation' does not
 */
__device__
d_Gslr_holder d_EvaluateGslr(double t, double f, cmplx *H, double* k, int response, double* p0){
    // response == 1 is full ,, response anything else is constellation
    //# Trajectories, p0 used only for the full response
    cmplx I(0.0, 1.0);
    cmplx m_I(0.0, -1.0);
    double alpha = Omega0*t; double c = cos(alpha); double s = sin(alpha);
    double a = aorbit; double e = eorbit;

    //double p0[3] = {a*c, a*s, 0.*t}; // funcp0(t)
    __shared__ double p1L_all[NUM_THREADS2 * 3];
    double* p1L = &p1L_all[threadIdx.x * 3];
    p1L[0] = - a*e*(1 + s*s);
    p1L[1] = a*e*c*s;
    p1L[2] = -a*e*sqrt3*c;

    __shared__ double p2L_all[NUM_THREADS2 * 3];
    double* p2L = &p2L_all[threadIdx.x * 3];
    p2L[0] = a*e/2*(sqrt3*c*s + (1 + s*s));
    p2L[1] = a*e/2*(-c*s - sqrt3*(1 + c*c));
    p2L[2] = -a*e*sqrt3/2*(sqrt3*s - c);

    __shared__ double p3L_all[NUM_THREADS2 * 3];
    double* p3L = &p3L_all[threadIdx.x * 3];
    p3L[0] = a*e/2*(-sqrt3*c*s + (1 + s*s));
    p3L[1] = a*e/2*(-c*s + sqrt3*(1 + c*c));
    p3L[2] = -a*e*sqrt3/2*(-sqrt3*s - c);

    __shared__ double n_all[NUM_THREADS2 * 3];
    double* n = &n_all[threadIdx.x * 3];

    // n1
    n[0] = -1./2*c*s;
    n[1] = 1./2*(1 + c*c);
    n[2] = sqrt3/2*s;

    double kn1= d_dot_product_1d(k, n);
    cmplx n1Hn1 = d_vec_H_vec_product(n, H, n); //np.dot(n1, np.dot(H, n1))

    // n2
    n[0] = c*s - sqrt3*(1 + s*s);
    n[1] = sqrt3*c*s - (1 + c*c);
    n[2] = -sqrt3*s - 3*c;

    for (int i=0; i<3; i++) n[i] = n[i]*1./4.;

    double kn2= d_dot_product_1d(k, n);
    cmplx n2Hn2 = d_vec_H_vec_product(n, H, n); //np.dot(n1, np.dot(H, n1))

    // n3

    n[0] = c*s + sqrt3*(1 + s*s);
    n[1] = -sqrt3*c*s - (1 + c*c);
    n[2] = -sqrt3*s + 3*c;

    for (int i=0; i<3; i++) n[i] = n[i]*1./4.;

    double kn3= d_dot_product_1d(k, n);
    cmplx n3Hn3 = d_vec_H_vec_product(n, H, n); //np.dot(n1, np.dot(H, n1))


    // # Compute intermediate scalar products
    // t scalar case

    double temp1 = p1L[0]+p2L[0]; double temp2 = p1L[1]+p2L[1]; double temp3 = p1L[2]+p2L[2];
    double temp4 = p2L[0]+p3L[0]; double temp5 = p2L[1]+p3L[1]; double temp6 = p2L[2]+p3L[2];
    double temp7 = p3L[0]+p1L[0]; double temp8 = p3L[1]+p1L[1]; double temp9 = p3L[2]+p1L[2];

    p1L[0] = temp1; p1L[1] = temp2; p1L[2] = temp3;  // now p1L_plus_p2L -> p1L
    p2L[0] = temp4; p2L[1] = temp5; p2L[2] = temp6;  // now p2L_plus_p3L -> p2L
    p3L[0] = temp7; p3L[1] = temp8; p3L[2] = temp9;  // now p3L_plus_p1L -> p3L

    double kp1Lp2L = d_dot_product_1d(k, p1L);
    double kp2Lp3L = d_dot_product_1d(k, p2L);
    double kp3Lp1L = d_dot_product_1d(k, p3L);
    double kp0 = d_dot_product_1d(k, p0);

    // # Prefactors - projections are either scalars or vectors
    cmplx factorcexp0;
    if (response==1) factorcexp0 = gcmplx::exp(I*2.*PI*f/C_SI * kp0); // I*2.*PI*f/C_SI * kp0
    else factorcexp0 = cmplx(1.0, 0.0);
    double prefactor = PI*f*L_SI/C_SI;

    cmplx factorcexp12 = gcmplx::exp(I*prefactor * (1.+kp1Lp2L/L_SI)); //prefactor * (1.+kp1Lp2L/L_SI)
    cmplx factorcexp23 = gcmplx::exp(I*prefactor * (1.+kp2Lp3L/L_SI)); //prefactor * (1.+kp2Lp3L/L_SI)
    cmplx factorcexp31 = gcmplx::exp(I*prefactor * (1.+kp3Lp1L/L_SI)); //prefactor * (1.+kp3Lp1L/L_SI)

    cmplx factorsinc12 = d_sinc( prefactor * (1.-kn3));
    cmplx factorsinc21 = d_sinc( prefactor * (1.+kn3));
    cmplx factorsinc23 = d_sinc( prefactor * (1.-kn1));
    cmplx factorsinc32 = d_sinc( prefactor * (1.+kn1));
    cmplx factorsinc31 = d_sinc( prefactor * (1.-kn2));
    cmplx factorsinc13 = d_sinc( prefactor * (1.+kn2));

    // # Compute the Gslr - either scalars or vectors
    d_Gslr_holder Gslr_out;


    cmplx commonfac = I*prefactor*factorcexp0;
    Gslr_out.G12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12;
    Gslr_out.G21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12;
    Gslr_out.G23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23;
    Gslr_out.G32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23;
    Gslr_out.G31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31;
    Gslr_out.G13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31;

    // ### FIXME
    // # G13 = -1j * prefactor * n2Hn2 * factorsinc31 * np.conjugate(factorcexp31)
    return Gslr_out;
}



__device__
d_transferL_holder d_TDICombinationFD(d_Gslr_holder Gslr, double f, int TDItag, int rescaled){
    // int TDItag == 1 is XYZ int TDItag == 2 is AET
    // int rescaled == 1 is True int rescaled == 0 is False
    d_transferL_holder transferL;
    cmplx factor, factorAE, factorT;
    cmplx I(0.0, 1.0);
    double x = PI*f*L_SI/C_SI;
    cmplx z = gcmplx::exp(I*2.*x);
    cmplx Xraw, Yraw, Zraw, Araw, Eraw, Traw;
    cmplx factor_convention, point5, c_one, c_two;
    if (TDItag==1){
        // # First-generation TDI XYZ
        // # With x=pifL, factor scaled out: 2I*sin2x*e2ix
        if (rescaled == 1) factor = 1.;
        else factor = 2.*I*sin(2.*x)*z;
        Xraw = Gslr.G21 + z*Gslr.G12 - Gslr.G31 - z*Gslr.G13;
        Yraw = Gslr.G32 + z*Gslr.G23 - Gslr.G12 - z*Gslr.G21;
        Zraw = Gslr.G13 + z*Gslr.G31 - Gslr.G23 - z*Gslr.G32;
        transferL.transferL1 = factor * Xraw;
        transferL.transferL2 = factor * Yraw;
        transferL.transferL3 = factor * Zraw;
        return transferL;
    }

    else{
        //# First-generation TDI AET from X,Y,Z
        //# With x=pifL, factors scaled out: A,E:I*sqrt2*sin2x*e2ix T:2*sqrt2*sin2x*sinx*e3ix
        //# Here we include a factor 2, because the code was first written using the definitions (2) of McWilliams&al_0911 where A,E,T are 1/2 of their LDC definitions
        factor_convention = cmplx(2.,0.0);
        if (rescaled == 1){
            factorAE = cmplx(1., 0.0);
            factorT = cmplx(1., 0.0);
        }
        else{
          factorAE = I*sqrt2*sin(2.*x)*z;
          factorT = 2.*sqrt2*sin(2.*x)*sin(x)*gcmplx::exp(I*3.*x);
        }

        Araw = 0.5 * ( (1.+z)*(Gslr.G31 + Gslr.G13) - Gslr.G23 - z*Gslr.G32 - Gslr.G21 - z*Gslr.G12 );
        Eraw = 0.5*invsqrt3 * ( (1.-z)*(Gslr.G13 - Gslr.G31) + (2.+z)*(Gslr.G12 - Gslr.G32) + (1.+2.*z)*(Gslr.G21 - Gslr.G23) );
        Traw = invsqrt6 * ( Gslr.G21 - Gslr.G12 + Gslr.G32 - Gslr.G23 + Gslr.G13 - Gslr.G31);
        transferL.transferL1 = factor_convention * factorAE * Araw;
        transferL.transferL2 = factor_convention * factorAE * Eraw;
        transferL.transferL3 = factor_convention * factorT * Traw;
        return transferL;
    }
}


__device__
d_transferL_holder d_JustLISAFDresponseTDI(cmplx *H, double f, double t, double lam, double beta, double t0, int TDItag, int order_fresnel_stencil){
    t = t + t0*YRSID_SI;

    //funck
    __shared__ double kvec_all[NUM_THREADS2 * 3];
    double* kvec = &kvec_all[threadIdx.x * 3];
    kvec[0] = -cos(beta)*cos(lam);
    kvec[1] = -cos(beta)*sin(lam);
    kvec[2] = -sin(beta);

    // funcp0
    double alpha = Omega0*t; double c = cos(alpha); double s = sin(alpha); double a = aorbit;

    __shared__ double p0_all[NUM_THREADS2 * 3];
    double* p0 = &p0_all[threadIdx.x * 3];
    p0[0] = a*c;
    p0[1] = a*s;
    p0[2] = 0.*t;

    // dot kvec with p0
    double kR = d_dot_product_1d(kvec, p0);

    double phaseRdelay = 2.*PI/clight *f*kR;

    // going to assume order_fresnel_stencil == 0 for now
    d_Gslr_holder Gslr = d_EvaluateGslr(t, f, H, kvec, 1, p0); // assumes full response
    d_Gslr_holder Tslr; // use same struct because its the same setup
    cmplx m_I(0.0, -1.0); // -1.0 -> mu_I

    // fill Tslr
    Tslr.G12 = Gslr.G12*gcmplx::exp(m_I*phaseRdelay); // really -I*
    Tslr.G21 = Gslr.G21*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G23 = Gslr.G23*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G32 = Gslr.G32*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G31 = Gslr.G31*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G13 = Gslr.G13*gcmplx::exp(m_I*phaseRdelay);

    d_transferL_holder transferL = d_TDICombinationFD(Tslr, f, TDItag, 0);
    transferL.phaseRdelay = phaseRdelay;
    return transferL;
}



 /**
  * Michael Katz added this function.
  * internal function that filles amplitude and phase for a specific frequency and mode.
  */
 __device__
 void calculate_modes(int binNum, int mode_i, double* amps, double* phases, double* phases_deriv, double* freqs, int ell, int mm, PhenomHMStorage *pHM, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double amp0, double Rholm, double Taulm, double t0, double phi0, int length, int numBinAll, int numModes, double M_tot_sec, double cshift[])
 {

         double amp_i, phase_i, dphidf, phase_up, phase_down;
         double t_wave_frame, t_sampling_frame;
         int status_in_for;
         UsefulPowers powers_of_f;
         int retcode = 0;
         double eps = 1e-9;

         for (int i = threadIdx.x; i < length; i += blockDim.x)
         {
             //int mode_index = (i * numModes + mode_i) * numBinAll + binNum;
             //int freq_index = i * numBinAll + binNum;

             int mode_index = (binNum * numModes + mode_i) * length + i;
             int freq_index = binNum * length + i;

             double freq = freqs[freq_index];
             double freq_geom = freq*M_tot_sec;

             int retcode = 0;

             get_amp(&amp_i, freq_geom, ell, mm, pHM, powers_of_f, pAmp, amp_prefactors, amp0);

             get_phase(&phase_i, freq_geom, ell, mm, pHM, powers_of_f, pDPreComp, q, cshift, Rholm, Taulm, t0, phi0);

             amps[mode_index] = amp_i;
             phases[mode_index] = phase_i;

             get_phase(&phase_up, freq_geom + 0.5 * eps, ell, mm, pHM, powers_of_f, pDPreComp, q, cshift, Rholm, Taulm, t0, phi0);
             get_phase(&phase_down, freq_geom - 0.5 * eps, ell, mm, pHM, powers_of_f, pDPreComp, q, cshift, Rholm, Taulm, t0, phi0);

             dphidf = (phase_up - phase_down)/eps;
             phases_deriv[mode_index] = dphidf;

              //t_wave_frame = 1./(2.0*PI)*dphidf + tRef_wave_frame;
              //t_sampling_frame = 1./(2.0*PI)*dphidf + tRef_sampling_frame;

              //d_transferL_holder transferL = d_JustLISAFDresponseTDI(H, freq, t_wave_frame, lam, beta, tBase, TDItag, order_fresnel_stencil);

         }

         /*
         double phasetimeshift;
         double phi_up, phi;

         double t, t_wave_frame, t_sampling_frame, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3, f_last, Shift, t_merger, dphidf, dphidf_merger;
         int old_ind_below;

         double eps = 1e-6;



                 if(i == num_points-1){
                   coeff_1 = mode_vals[mode_index].phase_coeff_1[num_points-2];
                   coeff_2 = mode_vals[mode_index].phase_coeff_2[num_points-2];
                   coeff_3 = mode_vals[mode_index].phase_coeff_3[num_points-2];

                   x = old_freqs[walker_i*num_points + i] - old_freqs[walker_i*num_points + (i-1)];
                   x2 = x*x;
                   dphidf = coeff_1 + 2.0*coeff_2*x + 3.0*coeff_3*x2;

                 } else{
                   dphidf = mode_vals[mode_index].phase_coeff_1[i];

                 }

                 t_wave_frame = 1./(2.0*PI)*dphidf + tRef_wave_frame;
                 t_sampling_frame = 1./(2.0*PI)*dphidf + tRef_sampling_frame;

                 // adjust phase values stored in mode vals to reflect the tRef shift
                 //mode_vals[mode_index].phase[i] += 2.0*PI*f*tRef_wave_frame;

                 d_transferL_holder transferL = d_JustLISAFDresponseTDI(&H[mode_index*9], f, t_wave_frame, lam, beta, t0, TDItag, order_fresnel_stencil);

                 mode_vals[mode_index].time_freq_corr[i] = t_sampling_frame + t0*YRSID_SI; // TODO: decide how to cutoff because it should be in terms of tL but it*/
}




 /**
  * Michael Katz added this function.
  * internal function that filles amplitude and phase for a specific frequency and mode.
  */
 __device__
 void response_modes(double* phases, double* response_out, int binNum, int mode_i, double* phases_deriv, double* freqs, double phiRef, int ell, int mm, int length, int numBinAll, int numModes,
 cmplx* H, double lam, double beta, double tRef_wave_frame, double tRef_sampling_frame, double tBase, int TDItag, int order_fresnel_stencil)
 {

         double amp_i, phase_i, dphidf, phase_up, phase_down;
         double t_wave_frame, t_sampling_frame;
         int status_in_for;
         UsefulPowers powers_of_f;
         int retcode = 0;
         double eps = 1e-9;
         int start_ind = 0;

         for (int i = threadIdx.x; i < length; i += blockDim.x)
         {
             //int mode_index = (i * numModes + mode_i) * numBinAll + binNum;
             //int freq_index = i * numBinAll + binNum;

             int mode_index = (binNum * numModes + mode_i) * length + i;
             int freq_index = binNum * length + i;

             double freq = freqs[freq_index];
             //double freq_geom = freq*M_tot_sec;

             dphidf = phases_deriv[mode_index];

             t_wave_frame = 1./(2.0*PI)*dphidf + tRef_wave_frame;
             t_sampling_frame = 1./(2.0*PI)*dphidf + tRef_sampling_frame;

             d_transferL_holder transferL = d_JustLISAFDresponseTDI(H, freq, t_wave_frame, lam, beta, tBase, TDItag, order_fresnel_stencil);

             // transferL1_re
             start_ind = 0 * numBinAll * numModes * length;
             int start_ind_old = start_ind;
             response_out[start_ind + mode_index] = gcmplx::real(transferL.transferL1);

             // transferL1_im
             start_ind = 1 * numBinAll * numModes * length;
             response_out[start_ind + mode_index] = gcmplx::imag(transferL.transferL1);

             // transferL1_re
             start_ind = 2 * numBinAll * numModes * length;
             response_out[start_ind + mode_index] = gcmplx::real(transferL.transferL2);

             // transferL1_re
             start_ind = 3 * numBinAll * numModes * length;
             response_out[start_ind + mode_index] = gcmplx::imag(transferL.transferL2);

             // transferL1_re
             start_ind = 4 * numBinAll * numModes * length;
             response_out[start_ind + mode_index] = gcmplx::real(transferL.transferL3);

             // transferL1_re
             start_ind = 5 * numBinAll * numModes * length;
             response_out[start_ind + mode_index] = gcmplx::imag(transferL.transferL3);

             // time_freq_corr update
             phases_deriv[mode_index] = t_sampling_frame + tBase * YRSID_SI;
             phases[mode_index] +=  transferL.phaseRdelay; // TODO: check this / I think I just need to remove it if phaseRdelay is exactly equal to (tRef_wave_frame * f) phase shift

         }
}



/*
Calculate spin weighted spherical harmonics
*/
__device__
cmplx SpinWeightedSphericalHarmonic(int s, int l, int m, double theta, double phi){
    // l=2
    double fac;
    if ((l==2) && (m==-2)) fac =  sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 - cos( theta ))*( 1.0 - cos( theta ));
    else if ((l==2) && (m==-1)) fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 - cos( theta ));
    else if ((l==2) && (m==1)) fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 + cos( theta ));
    else if ((l==2) && (m==2)) fac =  sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 + cos( theta ))*( 1.0 + cos( theta ));
    // l=3
    else if ((l==3) && (m==-3)) fac =  sqrt(21.0/(2.0*PI))*cos(theta/2.0)*pow(sin(theta/2.0),5.0);
    else if ((l==3) && (m==-2)) fac =  sqrt(7.0/(4.0*PI))*(2.0 + 3.0*cos(theta))*pow(sin(theta/2.0),4.0);
    else if ((l==3) && (m==2)) fac =  sqrt(7.0/PI)*pow(cos(theta/2.0),4.0)*(-2.0 + 3.0*cos(theta))/2.0;
    else if ((l==3) && (m==3)) fac =  -sqrt(21.0/(2.0*PI))*pow(cos(theta/2.0),5.0)*sin(theta/2.0);
    // l=4
    else if ((l==4) && (m==-4)) fac =  3.0*sqrt(7.0/PI)*pow(cos(theta/2.0),2.0)*pow(sin(theta/2.0),6.0);
    else if ((l==4) && (m==-3)) fac =  3.0*sqrt(7.0/(2.0*PI))*cos(theta/2.0)*(1.0 + 2.0*cos(theta))*pow(sin(theta/2.0),5.0);

    else if ((l==4) && (m==3)) fac =  -3.0*sqrt(7.0/(2.0*PI))*pow(cos(theta/2.0),5.0)*(-1.0 + 2.0*cos(theta))*sin(theta/2.0);
    else if ((l==4) && (m==4)) fac =  3.0*sqrt(7.0/PI)*pow(cos(theta/2.0),6.0)*pow(sin(theta/2.0),2.0);

    // Result
    cmplx I(0.0, 1.0);
    if (m==0) return cmplx(fac, 0.0);
    else {
        cmplx phaseTerm(m*phi, 0.0);
        return fac * exp(I*phaseTerm);
    }
}



/*
custom dot product in 2d
*/
__device__
void dot_product_2d(double* out, double* arr1, int m1, int n1, double* arr2, int m2, int n2, int dev, int stride){

    // dev and stride are on output
    for (int i=0; i<m1; i++){
        for (int j=0; j<n2; j++){
            out[stride*(i * 3  + j) + dev] = 0.0;
            for (int k=0; k<n1; k++){
                out[stride*(i * 3  + j) + dev] += arr1[i * 3 + k]*arr2[k * 3 + j];
            }
        }
    }
}

/*
Custom dot product in 1d
*/
__device__
double dot_product_1d(double arr1[3], double arr2[3]){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}


/**
 * Michael Katz added this function.
 * Main function for calculating PhenomHM in the form used by Michael Katz
 * This is setup to allow for pre-allocation of arrays. Therefore, all arrays
 * should be setup outside of this function.
 */
__device__
void IMRPhenomHMCore(
    int *ells,
    int *mms,
    double* amps,
    double* phases,
    double* phases_deriv,
    double* freqs,                      /**< GW frequecny list [Hz] */
    double m1_SI,                               /**< primary mass [kg] */
    double m2_SI,                               /**< secondary mass [kg] */
    double chi1z,                               /**< aligned spin of primary */
    double chi2z,                               /**< aligned spin of secondary */
    const double distance,                      /**< distance [m] */
    const double phiRef,                        /**< orbital phase at f_ref */
    double f_ref,
    int length,                              /**< reference GW frequency */
    int numModes,
    int binNum,
    int numBinAll,
    double cshift[]
)
{


    double t0, amp0, phi0;
    /* setup PhenomHM model storage struct / structs */
    /* Compute quantities/parameters related to PhenomD only once and store them */
    //PhenomHMStorage *pHM;
    PhenomHMStorage pHMtemp;
    PhenomHMStorage* pHM = &pHMtemp;
    init_PhenomHM_Storage(
        pHM,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        f_ref,
        phiRef
    );


    /* populate the ringdown frequency array */
    /* If you want to model a new mode then you have to add it here. */
    /* (l,m) = (2,2) */

    IMRPhenomHMGetRingdownFrequency(
        &pHM->Mf_RD_22,
        &pHM->Mf_DM_22,
        2, 2,
        pHM->finmass, pHM->finspin);



    /* (l,m) = (2,2) */
    int ell, mm;
    ell = 2;
    mm = 2;
    pHM->Rho22 = 1.0;
    pHM->Tau22 = 1.0;


    // Prepare 22 coefficients
    PhenDAmpAndPhasePreComp pDPreComp22;
    int retcode = IMRPhenomDSetupAmpAndPhaseCoefficients(
        &pDPreComp22,
        pHM->m1,
        pHM->m2,
        pHM->chi1z,
        pHM->chi2z,
        pHM->Rho22,
        pHM->Tau22,
        pHM->Mf_RD_22,
        pHM->Mf_DM_22
    );

    // set f_ref to f_max

    //if (pHM->f_ref == 0.0){
        pHM->Mf_ref = pDPreComp22.pAmp.fmaxCalc;
        pHM->f_ref = PhenomUtilsMftoHz(pHM->Mf_ref, pHM->Mtot);
        //printf("%e, %e\n", pHM->f_ref, pHM->Mf_ref);
    //}

    /* compute the reference phase shift need to align the waveform so that
     the phase is equal to phiRef at the reference frequency f_ref. */
    /* the phase shift is computed by evaluating the phase of the
    (l,m)=(2,2) mode.
    phi0 is the correction we need to add to each mode. */
    double phiRef_to_zero = 0.0;
    double phi_22_at_f_ref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,  1.0, 1.0);

    // phi0 is passed into this function as a pointer.This is for compatibility with GPU.
    phi0 = 0.5 * (phi_22_at_f_ref + phiRef_to_zero); // TODO: check this, I think it should be half of phiRef as well

    // t0 is passed into this function as a pointer.This is for compatibility with GPU.
    t0 = IMRPhenomDComputet0(pHM->eta, pHM->chi1z, pHM->chi2z, pHM->finspin, &(pDPreComp22.pPhi), &(pDPreComp22.pAmp));

    // setup PhenomD info. Sub here is due to preallocated struct

    retcode = 0;

    const double Mtot = (m1_SI + m2_SI) / MSUN_SI;

   /* Compute the amplitude pre-factor */
   // amp0 is passed into this function as a pointer.This is for compatibility with GPU.
   amp0 = PhenomUtilsFDamp0(Mtot, distance); // TODO check if this is right units

    //HMPhasePreComp q;

    // prep q and pDPreComp for each mode in the loop below
    HMPhasePreComp qlm;
    PhenDAmpAndPhasePreComp pDPreComplm;

    double Rholm, Taulm;

    double M_tot_sec = (pHM->m1 + pHM->m2)*MTSUN_SI;

    for (int mode_i=0; mode_i<numModes; mode_i++){
        ell = ells[mode_i];
        mm = mms[mode_i];

        IMRPhenomHMGetRingdownFrequency(
            &pHM->Mf_RD_lm,
            &pHM->Mf_DM_lm,
            ell, mm,
            pHM->finmass, pHM->finspin);

        pHM->Rholm = pHM->Mf_RD_22 / pHM->Mf_RD_lm;
        pHM->Taulm = pHM->Mf_DM_lm / pHM->Mf_DM_22;

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
            Taulm,
            pHM->Mf_RD_lm,
            pHM->Mf_DM_lm);

        retcode = IMRPhenomHMPhasePreComp(&qlm, ell, mm, pHM, pDPreComplm);

        calculate_modes(binNum, mode_i, amps, phases, phases_deriv, freqs, ell, mm, pHM, &(pDPreComplm.pAmp), pDPreComplm.amp_prefactors, pDPreComplm, qlm, amp0, Rholm, Taulm, t0, phi0, length, numBinAll, numModes, M_tot_sec, cshift);

    }
}


/**
 * Michael Katz added this function.
 * Main function for calculating PhenomHM in the form used by Michael Katz
 * This is setup to allow for pre-allocation of arrays. Therefore, all arrays
 * should be setup outside of this function.
 */
__device__
void responseCore(
    double* phases,
    double* response_out,
    int *ells,
    int *mms,
    double* phases_deriv,
    double* freqs,                      /**< GW frequecny list [Hz] */
    const double phiRef,                        /**< orbital phase at f_ref */
    double f_ref,
    double inc,
    double lam,
    double beta,
    double psi,
    double tRef_wave_frame,
    double tRef_sampling_frame,
    int length,                              /**< reference GW frequency */
    int numModes,
    int binNum,
    int numBinAll,
    double tBase, int TDItag, int order_fresnel_stencil
)
{

    int ell, mm;

    //// setup response
    __shared__ double HSplus[9];
    __shared__ double HScross[9];

    if (threadIdx.x == 0)
    {
        HSplus[0] = 1.;
        HSplus[1] = 0.;
        HSplus[2] = 0.;
        HSplus[3] = 0.;
        HSplus[4] = -1.;
        HSplus[5] = 0.;
        HSplus[6] = 0.;
        HSplus[7] = 0.;
        HSplus[8] = 0.;

        HScross[0] = 0.;
        HScross[1] = 1.;
        HScross[2] = 0.;
        HScross[3] = 1.;
        HScross[4] = 0.;
        HScross[5] = 0.;
        HScross[6] = 0.;
        HScross[7] = 0.;
        HScross[8] = 0.;
    }
    __syncthreads();

    __shared__ cmplx H_mat_all[NUM_THREADS2 * 3 * 3];
    cmplx* H_mat = &H_mat_all[threadIdx.x * 3 * 3];

    //##### Based on the f-n by Sylvain   #####
    //__shared__ double Hplus_all[NUM_THREADS2 * 3 * 3];
    //__shared__ double Hcross_all[NUM_THREADS2 * 3 * 3];
    //double* Hplus = &Hplus_all[threadIdx.x * 3 * 3];
    //double* Hcross = &Hcross_all[threadIdx.x * 3 * 3];

    double* Htemp = (double*) &H_mat[0];  // Htemp alternates with Hplus and Hcross in order to save shared memory: Hp[0], Hc[0], Hp[1], Hc1]
    // Htemp is then transformed into H_mat

    // Wave unit vector
    __shared__ double kvec_all[NUM_THREADS2 * 3];
    double* kvec = &kvec_all[threadIdx.x * 3];
    kvec[0] = -cos(beta)*cos(lam);
    kvec[1] = -cos(beta)*sin(lam);
    kvec[2] = -sin(beta);

    // Compute constant matrices Hplus and Hcross in the SSB frame
    double clambd = cos(lam); double slambd = sin(lam);
    double cbeta = cos(beta); double sbeta = sin(beta);
    double cpsi = cos(psi); double spsi = sin(psi);

    __shared__ double O1_all[NUM_THREADS2 * 3 * 3];
    double* O1 = &O1_all[threadIdx.x * 3 * 3];
    O1[0] = cpsi*slambd-clambd*sbeta*spsi;
    O1[1] = -clambd*cpsi*sbeta-slambd*spsi;
    O1[2] = -cbeta*clambd;
    O1[3] = -clambd*cpsi-sbeta*slambd*spsi;
    O1[4] = -cpsi*sbeta*slambd+clambd*spsi;
    O1[5] = -cbeta*slambd;
    O1[6] = cbeta*spsi;
    O1[7] = cbeta*cpsi;
    O1[8] = -sbeta;

    __shared__ double invO1_all[NUM_THREADS2 * 3 * 3];
    double* invO1 = &invO1_all[threadIdx.x * 3 * 3];;
    invO1[0] = cpsi*slambd-clambd*sbeta*spsi;
    invO1[1] = -clambd*cpsi-sbeta*slambd*spsi;
    invO1[2] = cbeta*spsi;
    invO1[3] = -clambd*cpsi*sbeta-slambd*spsi;
    invO1[4] = -cpsi*sbeta*slambd+clambd*spsi;
    invO1[5] = cbeta*cpsi;
    invO1[6] = -cbeta*clambd;
    invO1[7] = -cbeta*slambd;
    invO1[8] = -sbeta;

    __shared__ double out1_all[NUM_THREADS2 * 3 * 3];

    double* out1 = &out1_all[threadIdx.x * 3 * 3];


    // get Hplus
    //if ((threadIdx.x + blockDim.x * blockIdx.x <= 1)) printf("INNER %d %e %e %e\n", threadIdx.x + blockDim.x * blockIdx.x, invO1[0], invO1[1], invO1[6]);

    dot_product_2d(out1, HSplus, 3, 3, invO1, 3, 3, 0, 1);

    dot_product_2d(Htemp, O1, 3, 3, out1, 3, 3, 0, 2);

    // get Hcross
    dot_product_2d(out1, HScross, 3, 3, invO1, 3, 3, 0, 1);
    dot_product_2d(Htemp, O1, 3, 3, out1, 3, 3, 1, 2);

    cmplx I = cmplx(0.0, 1.0);
    cmplx Ylm, Yl_m, Yfactorplus, Yfactorcross;

    cmplx trans1, trans2;

    for (int mode_i=0; mode_i<numModes; mode_i++){
        ell = ells[mode_i];
        mm = mms[mode_i];

        Ylm = SpinWeightedSphericalHarmonic(-2, ell, mm, inc, phiRef);
        Yl_m = pow(-1.0, ell)*gcmplx::conj(SpinWeightedSphericalHarmonic(-2, ell, -1*mm, inc, phiRef));
        Yfactorplus = 1./2 * (Ylm + Yl_m);
        //# Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, should be for correct phase conventions
        Yfactorcross = 1./2. * I * (Ylm - Yl_m); //  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# Yfactorcross = -1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# The matrix H_mat is now complex

        //# H_mat = np.conjugate((Yfactorplus*Hplus + Yfactorcross*Hcross))  ### SB: H_ij = H_mat A_22 exp(i\Psi(f))
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                trans1 = Htemp[2*(i * 3 + j) + 0];
                trans2 = Htemp[2*(i * 3 + j) + 1];
                H_mat[(i * 3 + j)] = (Yfactorplus*trans1+ Yfactorcross*trans2);
                //printf("(%d, %d): %e, %e\n", i, j, Hplus[i][j], Hcross[i][j]);
            }
        }

        response_modes(phases, response_out, binNum, mode_i, phases_deriv, freqs, phiRef, ell, mm, length, numBinAll, numModes,
        H_mat, lam, beta, tRef_wave_frame, tRef_sampling_frame, tBase, TDItag, order_fresnel_stencil);

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
     double* amps, /**< [out] Frequency-domain waveform hx */
     double* phases,
     double* phases_deriv,
     int* ells_in,
     int* mms_in,
     double* freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
     double* m1_SI,                        /**< mass of companion 1 (kg) */
     double* m2_SI,                        /**< mass of companion 2 (kg) */
     double* chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
     double* chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
     double* distance,               /**< distance of source (m) */
     double* phiRef,                 /**< reference orbital phase (rad) */
     double* f_ref,                        /**< Reference frequency */
     int numModes,
     int length,
     int numBinAll
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
    __shared__ int mms[MAX_MODES];

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

    for (int i = threadIdx.x; i < numModes; i += blockDim.x)
    {
        ells[i] = ells_in[i];
        mms[i] = mms_in[i];
    }

    __syncthreads();

    int binNum = blockIdx.x; // threadIdx.x + blockDim.x * blockIdx.x;

    if (binNum < numBinAll)
    {
        IMRPhenomHMCore(ells, mms, amps, phases, phases_deriv, freqs, m1_SI[binNum], m2_SI[binNum], chi1z[binNum], chi2z[binNum], distance[binNum], phiRef[binNum], f_ref[binNum], length, numModes, binNum, numBinAll, cShift);
    }
}



////////////
// response
////////////



 CUDA_KERNEL
 void response(
     double* phases,
     double* response_out,
     double* phases_deriv,
     int* ells_in,
     int* mms_in,
     double* freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
     double* phiRef,                 /**< reference orbital phase (rad) */
     double* f_ref,                        /**< Reference frequency */
     double* inc,
     double* lam,
     double* beta,
     double* psi,
     double* tRef_wave_frame,
     double* tRef_sampling_frame,
     double tBase, int TDItag, int order_fresnel_stencil,
     int numModes,
     int length,
     int numBinAll
)
{

    __shared__ int ells[MAX_MODES];
    __shared__ int mms[MAX_MODES];

    for (int i = threadIdx.x; i < numModes; i += blockDim.x)
    {
        ells[i] = ells_in[i];
        mms[i] = mms_in[i];
    }

    __syncthreads();

    int binNum = blockIdx.x; // threadIdx.x + blockDim.x * blockIdx.x;

    if (binNum < numBinAll)
    {
        responseCore(phases, response_out, ells, mms, phases_deriv, freqs, phiRef[binNum], f_ref[binNum], inc[binNum], lam[binNum], beta[binNum], psi[binNum], tRef_wave_frame[binNum], tRef_sampling_frame[binNum], length, numModes, binNum, numBinAll,
        tBase, TDItag, order_fresnel_stencil);
    }
}


__device__
void prep_splines(int i, int length, int interp_i, int ninterps, int num_intermediates, double *b, double *ud, double *diag, double *ld, double *x, double *y, int numBinAll, int param, int nsub, int sub_i){
  double dx1, dx2, d, slope1, slope2;
  int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

  double xval0, xval1, xval2, yval1;

  int numFreqarrs = int(ninterps / num_intermediates);
  int freqArr_i = int(interp_i / num_intermediates);

  //if ((threadIdx.x == 10) && (blockIdx.x == 1)) printf("numFreqarrs %d %d %d %d %d\n", ninterps, interp_i, num_intermediates, numFreqarrs, freqArr_i);
  if (i == length - 1){
    ind0y = (param * nsub + sub_i) * length + (length - 3);
    ind1y = (param * nsub + sub_i) * length + (length - 2);
    ind2y = (param * nsub + sub_i) * length + (length - 1);

    ind0x = freqArr_i * length + (length - 3);
    ind1x = freqArr_i * length + (length - 2);
    ind2x = freqArr_i * length + (length - 1);

    ind_out = (param * nsub + sub_i) * length + (length - 1);

    xval0 = x[ind0x];
    xval1 = x[ind1x];
    xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;
    d = xval2 - xval0;

    yval1 = y[ind1y];

    slope1 = (yval1 - y[ind0y])/dx1;
    slope2 = (y[ind2y] - yval1)/dx2;

    b[ind_out] = ((dx2*dx2*slope1 +
                             (2*d + dx2)*dx1*slope2) / d);
    diag[ind_out] = dx1;
    ld[ind_out] = d;
    ud[ind_out] = 0.0;

  } else if (i == 0){

      ind0y = (param * nsub + sub_i) * length + 0;
      ind1y = (param * nsub + sub_i) * length + 1;
      ind2y = (param * nsub + sub_i) * length + 2;

      ind0x = freqArr_i * length + 0;
      ind1x = freqArr_i * length + 1;
      ind2x = freqArr_i * length + 2;

      ind_out = (param * nsub + sub_i) * length + 0;

      xval0 = x[ind0x];
      xval1 = x[ind1x];
      xval2 = x[ind2x];


      dx1 = xval1 - xval0;
      dx2 = xval2 - xval1;
      d = xval2 - xval0;

      yval1 = y[ind1y];

      //amp
      slope1 = (yval1 - y[ind0y])/dx1;
      slope2 = (y[ind2y] - yval1)/dx2;

      b[ind_out] = ((dx1 + 2*d) * dx2 * slope1 +
                          dx1*dx1 * slope2) / d;
    ud[ind_out] = d;
    ld[ind_out] = 0.0;
      diag[ind_out] = dx2;

  } else{

      ind0y = (param * nsub + sub_i) * length + (i - 1);
      ind1y = (param * nsub + sub_i) * length + (i + 0);
      ind2y = (param * nsub + sub_i) * length + (i + 1);

      ind0x = freqArr_i * length + (i - 1);
      ind1x = freqArr_i * length + (i - 0);
      ind2x = freqArr_i * length + (i + 1);

      ind_out = (param * nsub + sub_i) * length + i;

      xval0 = x[ind0x];
      xval1 = x[ind1x];
      xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;

    yval1 = y[ind1y];

    //amp
    slope1 = (yval1 - y[ind0y])/dx1;
    slope2 = (y[ind2y] - yval1)/dx2;

    b[ind_out] = 3.0* (dx2*slope1 + dx1*slope2);
    diag[ind_out] = 2*(dx1 + dx2);
    ud[ind_out] = dx1;
    ld[ind_out] = dx2;
  }

}



CUDA_KERNEL
void fill_B(double *freqs_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
                      int ninterps, int length, int num_intermediates, int numModes, int numBinAll){

    int param = 0;
    int nsub = 0;
    int sub_i = 0;
    #ifdef __CUDACC__

    int start1 = blockIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x;

    #else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

    #endif
    for (int interp_i = start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

         #ifdef __CUDACC__

         int start2 = threadIdx.x;
         int end2 = length;
         int diff2 = blockDim.x;

         #else

         int start2 = 0;
         int end2 = length;
         int diff2 = 1;

         #endif

        param = int((double) interp_i/(numModes * numBinAll));
        nsub = numModes * numBinAll;
        sub_i = interp_i % (numModes * numBinAll);

       for (int i = start2;
            i < end2;
            i += diff2){

            int lead_ind = interp_i*length;
            prep_splines(i, length, interp_i, ninterps, num_intermediates, B, upper_diag, diag, lower_diag, freqs_arr, y_all, numBinAll, param, nsub, sub_i);

}
}
}

/*
CuSparse error checking
*/
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)

#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

void interpolate_kern(int m, int n, double *a, double *b, double *c, double *d_in)
{
        #ifdef __CUDACC__
        size_t bufferSizeInBytes;

        cusparseHandle_t handle;
        void *pBuffer;

        CUSPARSE_CALL(cusparseCreate(&handle));
        CUSPARSE_CALL( cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
        gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

        CUSPARSE_CALL(cusparseDgtsv2StridedBatch(handle,
                                                  m,
                                                  a, // dl
                                                  b, //diag
                                                  c, // du
                                                  d_in,
                                                  n,
                                                  m,
                                                  pBuffer));

      CUSPARSE_CALL(cusparseDestroy(handle));
      gpuErrchk(cudaFree(pBuffer));

      #else

    #ifdef __USE_OMP__
    #pragma omp parallel for
    #endif
    for (int j = 0;
         j < n;
         j += 1){
           //fit_constants_serial(m, n, w, a, b, c, d_in, x_in, j);
           int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j*m + 1], &b[j*m], &c[j*m], &d_in[j*m], m);
           //if (info != m) printf("lapack info check: %d\n", info);

       }

      #endif


    /*
    int interp_i = threadIdx.x + blockDim.x * blockIdx.x;

    int param = (int) (interp_i / (numModes * numBinAll));
    int nsub = numBinAll * numModes;
    int sub_i = interp_i % (numModes * numBinAll);

    int ind_i, ind_im1, ind_ip1;
    if (interp_i < ninterps)
    {

        double w = 0.0;
        for (int i = 1; i < n; i += 1)
        {
            ind_i = (param * n + i) * nsub + sub_i;
            ind_im1 = (param * n + (i-1)) * nsub + sub_i;


            w = a[ind_i]/b[ind_im1];
            b[ind_i] = b[ind_i] - w * c[ind_im1];
            d[ind_i] = d[ind_i] - w * d[ind_im1];
        }

        ind_i = (param * n + (n-1)) * nsub + sub_i;

        d[ind_i] = d[ind_i]/b[ind_i];
        for (int i = n - 2; i >= 0; i -= 1)
        {
            ind_i = (param * n + i) * nsub + sub_i;
            ind_ip1 = (param * n + (i+1)) * nsub + sub_i;

            d[ind_i] = (d[ind_i] - c[ind_i] * d[ind_ip1])/b[ind_i];

        }
    }
    */
}


CUDA_CALLABLE_MEMBER
void fill_coefficients(int i, int length, int sub_i, int nsub, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3, int param){
  double slope, t, dydx_i;

  int ind_i = (param * nsub + sub_i) * length + i;
  int ind_ip1 = (param * nsub + sub_i) * length + (i + 1);

  slope = (y[ind_ip1] - y[ind_i])/dx;

  dydx_i = dydx[ind_i];

  t = (dydx_i + dydx[ind_ip1] - 2*slope)/dx;

  coeff1[ind_i] = dydx_i;
  coeff2[ind_i] = (slope - dydx_i) / dx - t;
  coeff3[ind_i] = t/dx;

  //if ((param == 1) && (i == length - 3) && (sub_i == 0)) printf("freq check: %d %d %d %d %d\n", i, dydx[ind_i], dydx[ind_ip1]);


}

CUDA_KERNEL
void set_spline_constants(double *f_arr, double* y, double *c1, double* c2, double* c3, double *B,
                      int ninterps, int length, int num_intermediates, int numBinAll, int numModes){

    double df;
    #ifdef __CUDACC__
    int start1 = blockIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x;
    #else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

    #endif

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

     int numFreqarrs = int(ninterps / num_intermediates);
     int freqArr_i = int(interp_i / num_intermediates);

     int param = (int) (interp_i / (numModes * numBinAll));
     int nsub = numBinAll * numModes;
     int sub_i = interp_i % (numModes * numBinAll);

     #ifdef __CUDACC__
     int start2 = threadIdx.x;
     int end2 = length - 1;
     int diff2 = blockDim.x;
     #else

     int start2 = 0;
     int end2 = length - 1;
     int diff2 = 1;

     #endif
     for (int i = start2;
            i < end2;
            i += diff2){

                // TODO: check if there is faster way to do this
              df = f_arr[freqArr_i * length + (i + 1)] - f_arr[freqArr_i * length + i];

              int lead_ind = interp_i*length;
              fill_coefficients(i, length, sub_i, nsub, B, df,
                                y,
                                c1,
                                c2,
                                c3, param);

}
}
}


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void atomicAddComplex(cmplx* a, cmplx b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAddDouble(x, b.real());
  atomicAddDouble(y, b.imag());
}



#define  DATA_BLOCK 128
#define  NUM_INTERPS 9

__device__
cmplx get_ampphasefactor(double amp, double phase, double phaseShift){
    return amp*gcmplx::exp(cmplx(0.0, phase + phaseShift));
}

__device__
cmplx combine_information(cmplx* channel1, cmplx* channel2, cmplx* channel3, double amp, double phase, double tf, cmplx transferL1, cmplx transferL2, cmplx transferL3, double t_start, double t_end)
{
    // TODO: make sure the end of the ringdown is included
    if ((tf >= t_start) && ((tf <= t_end) || (t_end <= 0.0)) && (amp > 1e-40))
    {
        cmplx amp_phase_term = amp*gcmplx::exp(cmplx(0.0, -phase));  // add phase shift

        *channel1 = gcmplx::conj(transferL1 * amp_phase_term);
        *channel2 = gcmplx::conj(transferL2 * amp_phase_term);
        *channel3 = gcmplx::conj(transferL3 * amp_phase_term);

    }
}

#define  NUM_TERMS 4

#define  MAX_NUM_COEFF_TERMS 1000

CUDA_KERNEL
void TDI(cmplx* templateChannels, double* dataFreqsIn, double dlog10f, double* freqsOld, double* propArrays, double* c1In, double* c2In, double* c3In, double t_mrg, int old_length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, int* inds, int ind_start, int ind_length, int bin_i)
{

    __shared__ double y[MAX_NUM_COEFF_TERMS];
    __shared__ double c1[MAX_NUM_COEFF_TERMS];
    __shared__ double c2[MAX_NUM_COEFF_TERMS];
    __shared__ double c3[MAX_NUM_COEFF_TERMS];
    __shared__ double freqs_shared[MAX_NUM_COEFF_TERMS];

    int num_params = 9;
    int mode_i = blockIdx.y;

    int numAll = numBinAll * numModes * old_length;

    double amp, phase, tfCorr, transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    double x, x2, x3, tempLike, addLike, time_check, phaseShift;
    cmplx trans_complex1, trans_complex2, trans_complex3, ampphasefactor;

    __shared__ int start_ind, end_ind;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool run = true;
    if (i >= ind_length) run = false;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        start_ind = inds[i];
    }
    int max_thread_num = (ind_length - blockDim.x*blockIdx.x > NUM_THREADS3) ? NUM_THREADS3 : ind_length - blockDim.x*blockIdx.x;

    if (threadIdx.x == max_thread_num - 1)
    {
        end_ind = inds[i];
    }

    __syncthreads();

    int num_windows = end_ind - start_ind + 1;

    int nsub = numModes * numBinAll;

    //if (run) printf("%d %d %d %d %d %d %d %d\n", max_thread_num, threadIdx.x, blockDim.x, NUM_THREADS3, i, ind_length, start_ind, end_ind);

    for (int j = threadIdx.x; j < num_windows; j += blockDim.x)
    {
        int window_i = j;

        int old_ind = start_ind + window_i;

        if ((old_ind < 0) || (old_ind >= old_length))
        {
            continue;
        }

        freqs_shared[window_i] = freqsOld[old_ind];
    }

    __syncthreads();

    for (int j = threadIdx.x; j < num_params * num_windows; j += blockDim.x)
    {
        int window_i = j % num_windows;
        int param_i = (int) (j / num_windows);

        int old_ind = start_ind + window_i;

        if ((old_ind < 0) || (old_ind >= old_length))
        {
            continue;
        }

        int ind = ((param_i * numBinAll + bin_i) * numModes + mode_i) * old_length + old_ind;
        int ind_shared = window_i * num_params + param_i;

        y[ind_shared] = propArrays[ind];
        c1[ind_shared] = c1In[ind];
        c2[ind_shared] = c2In[ind];
        c3[ind_shared] = c3In[ind];

    }

    __syncthreads();

    if (run)
    {
        double f = dataFreqsIn[i + ind_start];

        int ind_here = inds[i];

        int window_i = ind_here - start_ind;

        double f_old = freqs_shared[window_i];

        double x = f - f_old;
        double x2 = x * x;
        double x3 = x * x2;

        int int_shared = window_i * num_params + 0;
        double amp = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 1;
        double phase = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 2;
        double tf = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 3;
        double transferL1_re = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 4;
        double transferL1_im = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 5;
        double transferL2_re = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 6;
        double transferL2_im = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 7;
        double transferL3_re = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 8;
        double transferL3_im = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        cmplx channel1(0.0, 0.0);
        cmplx channel2(0.0, 0.0);
        cmplx channel3(0.0, 0.0);

        combine_information(&channel1, &channel2, &channel3, amp, phase, tf, cmplx(transferL1_re, transferL1_im), cmplx(transferL2_re, transferL2_im), cmplx(transferL3_re, transferL3_im), t_obs_start, t_obs_end);

        atomicAddComplex(&templateChannels[0 * ind_length + i], channel1);
        atomicAddComplex(&templateChannels[1 * ind_length + i], channel2);
        atomicAddComplex(&templateChannels[2 * ind_length + i], channel3);
        //if ((mode_i == 0)) printf("%d %e %e %e %.18e\n", i, tRef_sampling_frame, tBase * YRSID_SI, tRef_sampling_frame + tBase * YRSID_SI, tf, start_ind, f, f_old, x, amp, y[int_shared], c1[int_shared], c2[int_shared], c3[int_shared]);

    }

}


void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg_in, double* t_start_in, double* t_end_in, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths)
{

    cudaStream_t streams[numBinAll];

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];
        int* inds = (int*) inds_ptrs[bin_i];

        double t_mrg = t_mrg_in[bin_i];
        double t_start = t_start_in[bin_i];
        double t_end = t_end_in[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        int nblocks3 = std::ceil((length_bin_i + NUM_THREADS3 -1)/NUM_THREADS3);
        cudaStreamCreate(&streams[bin_i]);

        dim3 gridDim(nblocks3, numModes);
        TDI<<<gridDim, NUM_THREADS3, 0, streams[bin_i]>>>(templateChannels, dataFreqs, dlog10f, freqs, propArrays, c1, c2, c3, t_mrg, length, data_length, numBinAll, numModes, t_start, t_end, inds, ind_start, length_bin_i, bin_i);

    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
}

#define  DATA_BLOCK2 512
CUDA_KERNEL
void hdynLikelihood(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn,
                    int numBinAll, int data_length, int nChannels)
{
    __shared__ cmplx A0temp[DATA_BLOCK2];
    __shared__ cmplx A1temp[DATA_BLOCK2];
    __shared__ cmplx B0temp[DATA_BLOCK2];
    __shared__ cmplx B1temp[DATA_BLOCK2];
    __shared__ double dataFreqs[DATA_BLOCK2];

    cmplx A0, A1, B0, B1;

    cmplx trans_complex(0.0, 0.0);
    cmplx prev_trans_complex(0.0, 0.0);
    double prevFreq = 0.0;
    double freq = 0.0;

    int currentStart = 0;

    cmplx r0, r1, r1Conj, tempLike1, tempLike2;
    double mag_r0, midFreq;

    int binNum = threadIdx.x + blockDim.x * blockIdx.x;

    if (true) // for (int binNum = threadIdx.x + blockDim.x * blockIdx.x; binNum < numBinAll; binNum += blockDim.x * gridDim.x)
    {
        tempLike1 = 0.0;
        tempLike2 = 0.0;
        for (int channel = 0; channel < nChannels; channel += 1)
        {
            prevFreq = 0.0;
            currentStart = 0;
            while (currentStart < data_length)
            {
                __syncthreads();
                for (int jj = threadIdx.x; jj < DATA_BLOCK2; jj += blockDim.x)
                {
                    if ((jj + currentStart) >= data_length) continue;
                    A0temp[jj] = dataConstants[(0 * nChannels + channel) * data_length + currentStart + jj];
                    A1temp[jj] = dataConstants[(1 * nChannels + channel) * data_length + currentStart + jj];
                    B0temp[jj] = dataConstants[(2 * nChannels + channel) * data_length + currentStart + jj];
                    B1temp[jj] = dataConstants[(3 * nChannels + channel) * data_length + currentStart + jj];

                    dataFreqs[jj] = dataFreqsIn[currentStart + jj];

                    //if ((jj + currentStart < 3) && (binNum == 0) & (channel == 0))
                    //    printf("check %e %e, %e %e, %e %e, %e %e, %e \n", A0temp[jj], A1temp[jj], B0temp[jj], B1temp[jj], dataFreqs[jj]);

                }
                __syncthreads();
                if (binNum < numBinAll)
                {
                    for (int jj = 0; jj < DATA_BLOCK2; jj += 1)
                    {
                        if ((jj + currentStart) >= data_length) continue;
                        freq = dataFreqs[jj];
                        trans_complex = templateChannels[((jj + currentStart) * nChannels + channel) * numBinAll + binNum];

                        if ((prevFreq != 0.0) && (jj + currentStart > 0))
                        {
                            A0 = A0temp[jj]; // constants will need to be aligned with 1..n-1 because there are data_length - 1 bins
                            A1 = A1temp[jj];
                            B0 = B0temp[jj];
                            B1 = B1temp[jj];

                            r1 = (trans_complex - prev_trans_complex)/(freq - prevFreq);
                            midFreq = (freq + prevFreq)/2.0;

                            r0 = trans_complex - r1 * (freq - midFreq);

                            //if (((binNum == 767) || (binNum == 768)) & (channel == 0))
                            //    printf("CHECK2: %d %d %d %e %e\n", jj + currentStart, binNum, jj, A0); // , %e %e, %e %e, %e %e, %e %e,  %e %e,  %e %e , %e\n", ind, binNum, jj + currentStart, A0, A1, B0, B1, freq, prevFreq, trans_complex, prev_trans_complex, midFreq);

                            r1Conj = gcmplx::conj(r1);

                            tempLike1 += A0 * gcmplx::conj(r0) + A1 * r1Conj;

                            mag_r0 = gcmplx::abs(r0);
                            tempLike2 += B0 * (mag_r0 * mag_r0) + 2. * B1 * gcmplx::real(r0 * r1Conj);
                        }

                        prev_trans_complex = trans_complex;
                        prevFreq = freq;
                    }
                }
                currentStart += DATA_BLOCK2;
            }
        }
        likeOut1[binNum] = tempLike1;
        likeOut2[binNum] = tempLike2;
    }
}





void LISA_response(
    double* response_out,
    int* ells_in,
    int* mms_in,
    double* freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
    double* phiRef,                 /**< reference orbital phase (rad) */
    double* f_ref,                        /**< Reference frequency */
    double* inc,
    double* lam,
    double* beta,
    double* psi,
    double* tRef_wave_frame,
    double* tRef_sampling_frame,
    double tBase, int TDItag, int order_fresnel_stencil,
    int numModes,
    int length,
    int numBinAll,
    int includesAmps
)
{

    int start_param = includesAmps;  // if it has amps, start_param is 1, else 0

    double* phases = &response_out[start_param * numBinAll * numModes * length];
    double* phases_deriv = &response_out[(start_param + 1) * numBinAll * numModes * length];
    double* response_vals = &response_out[(start_param + 2) * numBinAll * numModes * length];

    int nblocks2 = numBinAll; //std::ceil((numBinAll + NUM_THREADS2 -1)/NUM_THREADS2);

    response<<<nblocks2, NUM_THREADS2>>>(
        phases,
        response_vals,
        phases_deriv,
        ells_in,
        mms_in,
        freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
        phiRef,                 /**< reference orbital phase (rad) */
        f_ref,                        /**< Reference frequency */
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        tBase, TDItag, order_fresnel_stencil,
        numModes,
        length,
        numBinAll
   );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}


void waveform_amp_phase(
    double* waveformOut,
    int* ells_in,
    int* mms_in,
    double* freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
    double* m1_SI,                        /**< mass of companion 1 (kg) */
    double* m2_SI,                        /**< mass of companion 2 (kg) */
    double* chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
    double* chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
    double* distance,               /**< distance of source (m) */
    double* phiRef,                 /**< reference orbital phase (rad) */
    double* f_ref,                        /**< Reference frequency */
    int numModes,
    int length,
    int numBinAll
)
{

    double* amps = &waveformOut[0];
    double* phases = &waveformOut[numBinAll * numModes * length];
    double* phases_deriv = &waveformOut[2 * numBinAll * numModes * length];

    //int nblocks = std::ceil((numBinAll + NUM_THREADS -1)/NUM_THREADS);
    int nblocks = numBinAll; //std::ceil((numBinAll + NUM_THREADS -1)/NUM_THREADS);
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);*/
        //printf("%d begin\n", jj);
    IMRPhenomHM<<<nblocks, NUM_THREADS>>>(
        amps, /**< [out] Frequency-domain waveform hx */
        phases,
        phases_deriv,
        ells_in,
        mms_in,
        freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
        m1_SI,                        /**< mass of companion 1 (kg) */
        m2_SI,                        /**< mass of companion 2 (kg) */
        chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
        chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
        distance,               /**< distance of source (m) */
        phiRef,                 /**< reference orbital phase (rad) */
        f_ref,                        /**< Reference frequency */
        numModes,
        length,
        numBinAll
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    /*
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%e\n", milliseconds);*/

}


void interpolate(double* freqs, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll)
{

    int num_intermediates = numModes * numInterpParams;
    int ninterps = numModes * numInterpParams * numBinAll;

    int nblocks = std::ceil((ninterps + NUM_THREADS -1)/NUM_THREADS);

    double* c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    double* c2 = diag; //&interp_array[1 * numInterpParams * amp_phase_size];
    double* c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];

    //printf("%d after response, %d\n", jj, nblocks2);

     fill_B<<<nblocks, NUM_THREADS>>>(freqs, propArrays, B, upper_diag, diag, lower_diag, ninterps, length, num_intermediates, numModes, numBinAll);
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

     //printf("%d after fill b\n", jj);
     interpolate_kern(length, ninterps, lower_diag, diag, upper_diag, B);


  set_spline_constants<<<nblocks, NUM_THREADS>>>(freqs, propArrays, c1, c2, c3, B,
                    ninterps, length, num_intermediates, numBinAll, numModes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    //printf("%d after set spline\n", jj);
}

CUDA_KERNEL
void fill_waveform(cmplx* channel1, cmplx* channel2, cmplx* channel3,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes)
{

    cmplx I(0.0, 1.0);

    cmplx temp_channel1 = 0.0, temp_channel2 = 0.0, temp_channel3 = 0.0;
    for (int binNum = threadIdx.x + blockDim.x * blockIdx.x; binNum < numBinAll; binNum += gridDim.x * blockDim.x)
    {
        for (int i = 0; i < data_length; i += 1)
        {

            temp_channel1 = 0.0 + 0.0 * I;
            temp_channel2 = 0.0 + 0.0 * I;
            temp_channel3 = 0.0 + 0.0 * I;

            for (int mode_i = 0; mode_i < numModes; mode_i += 1)
            {
                int ind = ((0 * data_length + i) * numModes + mode_i) * numBinAll + binNum;
                double amp = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double phase = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                //double phase_deriv = bb_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double transferL1_re = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double transferL1_im = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double transferL2_re = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double transferL2_im = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double transferL3_re = bbh_buffer[ind];

                ind += data_length * numModes * numBinAll;
                double transferL3_im = bbh_buffer[ind];

                cmplx amp_phase = amp * gcmplx::exp(-I * phase);

                temp_channel1 += amp * cmplx(transferL1_re, transferL1_im);
                temp_channel2 += amp * cmplx(transferL2_re, transferL2_im);
                temp_channel3 += amp * cmplx(transferL3_re, transferL3_im);

            }

            channel1[i * numBinAll + binNum] = temp_channel1;
            channel2[i * numBinAll + binNum] = temp_channel2;
            channel3[i * numBinAll + binNum] = temp_channel3;

        }
    }
}

void direct_sum(cmplx* channel1, cmplx* channel2, cmplx* channel3,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes)
{

    int nblocks5 = std::ceil((numBinAll + NUM_THREADS4 -1)/NUM_THREADS4);

    fill_waveform<<<nblocks5, NUM_THREADS4>>>(channel1, channel2, channel3, bbh_buffer, numBinAll, data_length, nChannels, numModes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels)
{

    int nblocks4 = std::ceil((numBinAll + NUM_THREADS4 -1)/NUM_THREADS4);

    hdynLikelihood<<<nblocks4, NUM_THREADS4>>>(likeOut1, likeOut2, templateChannels, dataConstants, dataFreqs, numBinAll, data_length, nChannels);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

/*
int main()
{

    int TDItag = 1;
    int order_fresnel_stencil = 0;
    double tBase = 1.0;

    int numBinAll = 5000;
    int numModes = 6;
    int length = 1024;
    int data_length = 4096;

    int *ells_in, *mms_in;

    gpuErrchk(cudaMallocManaged(&ells_in, numModes * sizeof(int)));
    gpuErrchk(cudaMallocManaged(&mms_in, numModes * sizeof(int)));

    ells_in[0] = 2;
    ells_in[1] = 3;
    ells_in[2] = 4;

    ells_in[3] = 2;
    ells_in[4] = 3;
    ells_in[5] = 4;

    mms_in[0] = 2;
    mms_in[1] = 3;
    mms_in[2] = 4;

    mms_in[3] = 1;
    mms_in[4] = 2;
    mms_in[5] = 3;

    double *amps, *phases, *phases_deriv, *freqs, *m1_SI, *m2_SI, *chi1z, *chi2z, *distance, *phiRef, *fRef;
    double *inc, *lam, *beta, *psi, *tRef_wave_frame, *tRef_sampling_frame;
    double *response_out;
    double *B, *interp_array; // plays roll of upper lower diag, and then coefficients 1, 2, 3

    size_t amp_phase_size = numBinAll * numModes * length *sizeof(double);
    size_t freqs_size = numBinAll * length * sizeof(double);
    size_t bin_size = numBinAll * sizeof(double);

    int numInterpParams = 9;

    gpuErrchk(cudaMallocManaged(&amps, numInterpParams * amp_phase_size));

    response_out = &amps[1 * numBinAll * numModes * length];

    double *upper_diag, *diag, *lower_diag;
    gpuErrchk(cudaMallocManaged(&B, numInterpParams * amp_phase_size));
    gpuErrchk(cudaMallocManaged(&upper_diag, numInterpParams * amp_phase_size));
    gpuErrchk(cudaMallocManaged(&diag, numInterpParams * amp_phase_size));
    gpuErrchk(cudaMallocManaged(&lower_diag, numInterpParams * amp_phase_size));

    //double* upper_diag = &interp_array[0 * numInterpParams * amp_phase_size];
    //double* diag = &interp_array[1 * numInterpParams * amp_phase_size];
    //double* lower_diag = &interp_array[2 * numInterpParams * amp_phase_size];

    double* propArrays = amps;

    gpuErrchk(cudaMallocManaged(&freqs, freqs_size));

    gpuErrchk(cudaMallocManaged(&m1_SI, bin_size));
    gpuErrchk(cudaMallocManaged(&m2_SI, bin_size));
    gpuErrchk(cudaMallocManaged(&chi1z, bin_size));
    gpuErrchk(cudaMallocManaged(&chi2z, bin_size));
    gpuErrchk(cudaMallocManaged(&distance, bin_size));
    gpuErrchk(cudaMallocManaged(&phiRef, bin_size));
    gpuErrchk(cudaMallocManaged(&fRef, bin_size));

    gpuErrchk(cudaMallocManaged(&inc, bin_size));
    gpuErrchk(cudaMallocManaged(&lam, bin_size));
    gpuErrchk(cudaMallocManaged(&beta, bin_size));
    gpuErrchk(cudaMallocManaged(&psi, bin_size));
    gpuErrchk(cudaMallocManaged(&tRef_wave_frame, bin_size));
    gpuErrchk(cudaMallocManaged(&tRef_sampling_frame, bin_size));

    double m1 = 2e6; // solar
    double m2 = 1e6;
    double a1 = 0.8;
    double a2 = 0.8;
    double dist = 30.0; // Gpc
    double phi_ref = 0.0;
    double f_ref = 0.0;
    double inc_in = PI/3.;
    double lam_in = 0.4;
    double beta_in = 0.24;
    double psi_in = 1.0;
    double tRef_wave_frame_in = 10.0;
    double tRef_sampling_frame_in = 50.0;

    double Msec = (m1 + m2) * MTSUN_SI;

    double log10f_start = log10(1e-4/Msec);
    double log10f_end = log10(0.6/Msec);

    double dlog10f = (log10f_end - log10f_start)/(length - 1);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        m1_SI[bin_i] = (1e6 * MSUN_SI) * (1 + distribution(generator));
        m2_SI[bin_i] = (4e5 * MSUN_SI) * (1 + distribution(generator));

        chi1z[bin_i] = (distribution(generator))* 0.9;
        chi2z[bin_i] = (distribution(generator))* 0.9;

        distance[bin_i] = (35) * (1 + distribution(generator)) * 1e9 * PC_SI;
        phiRef[bin_i] = (1 + distribution(generator));
        fRef[bin_i] = f_ref;

        inc[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        lam[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        beta[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        psi[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        tRef_wave_frame[bin_i] = (1 + distribution(generator)) * 20.0;
        tRef_sampling_frame[bin_i] = (1 + distribution(generator)) * 20.0;

        for (int i = 0; i < length; i += 1)
        {
            freqs[i * numBinAll + bin_i] = pow(10.0, log10f_start + i * dlog10f);
        }
    }

    cmplx *dataChannels, *templateChannels, *dataConstants;
    double *dataFreqs;
    int nChannels = 3;

    double t_obs_start = 1.0;
    double t_obs_end = 0.0;

    gpuErrchk(cudaMallocManaged(&dataChannels, nChannels * data_length * sizeof(cmplx)));
    gpuErrchk(cudaMallocManaged(&dataConstants, NUM_TERMS * nChannels * data_length * sizeof(cmplx)));
    gpuErrchk(cudaMallocManaged(&templateChannels, numBinAll * nChannels * data_length * sizeof(cmplx)));
    gpuErrchk(cudaMallocManaged(&dataFreqs, data_length * sizeof(double)));

    double dlog10fData = (log10f_end - log10f_start)/(data_length - 1);

    for (int i = 0; i < data_length; i += 1)
    {
        dataFreqs[i] = pow(10.0, log10f_start + i * dlog10fData);

        for (int channel = 0; channel < nChannels; channel += 1)
        {
            dataChannels[channel * data_length + i] = cmplx(1.0, 1.0);

            for (int constant = 0; constant < NUM_TERMS; constant += 1)
            {
                dataConstants[(constant * nChannels + channel) * data_length + i] = cmplx(1.0, 1.0);
            }
        }
    }

    cmplx *likeOut1;
    gpuErrchk(cudaMallocManaged(&likeOut1, numBinAll * sizeof(cmplx)));

    cmplx *likeOut2;
    gpuErrchk(cudaMallocManaged(&likeOut2, numBinAll * sizeof(cmplx)));

    double *c1, *c2, *c3;
    int numIter = 10;

    for (int jj = 0; jj < numIter; jj += 1)
    {

        //printf("%d begin\n", jj);
        waveform_amp_phase(
        amps, ///**< [out] Frequency-domain waveform hx
        ells_in,
        mms_in,
        freqs,               ///**< Frequency points at which to evaluate the waveform (Hz)
        m1_SI,                       // /**< mass of companion 1 (kg)
        m2_SI,                        ///**< mass of companion 2 (kg)
        chi1z,                        ///**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
        chi2z,                        ///**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)
        distance,               ///**< distance of source (m)
        phiRef,                 ///**< reference orbital phase (rad)
        fRef,                      //  /**< Reference frequency
        numModes,
        length,
        numBinAll
   );

   int includesAmps = 0;
   LISA_response(
       response_out,
       ells_in,
       mms_in,
       freqs,               ///**< Frequency points at which to evaluate the waveform (Hz)
       phiRef,                // /**< reference orbital phase (rad)
       fRef,                    //    /**< Reference frequency
       inc,
       lam,
       beta,
       psi,
       tRef_wave_frame,
       tRef_sampling_frame,
       tBase, TDItag, order_fresnel_stencil,
       numModes,
       length,
       numBinAll,
       includesAmps
  );

  interpolate(freqs, propArrays,
                   B, upper_diag, diag, lower_diag,
                 length, numInterpParams, numModes, numBinAll);

    //printf("%d middle\n", jj);

    c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    c2 = diag; //&interp_array[1 * numInterpParams * amp_phase_size];
    c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];


    InterpTDI(templateChannels, dataChannels, dataFreqs, freqs, propArrays, c1, c2, c3, tBase, tRef_sampling_frame, tRef_wave_frame, length, data_length,   numBinAll, numModes, t_obs_start, t_obs_end);

    hdyn(likeOut1, likeOut2, templateChannels, dataConstants, dataFreqs, numBinAll, data_length, nChannels);
    }

    int binNum = 1000;
    int mode_i = 0;
    for (int i = 0; i < 5; i += 1) printf("%d %e %e\n", i, c1[(i * numModes + 0) * numBinAll + 0], c2[(i * numModes + 0) * numBinAll + 0]);

    return 0;
}

*/

/*
__device__
void fill_coefficients(int i, int length, int mode_i, int numModes, int interp_i, int ninterps, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3){
  double slope, t, dydx_i;

  int indip1 = ((i + 1) * numModes + mode_i) * ninterps + interp_i;
  int indi = ((i) * numModes + mode_i) * ninterps + interp_i;

  slope = (y[indip1] - y[indi])/dx;

  dydx_i = dydx[indi];

  t = (dydx_i + dydx[indip1] - 2*slope)/dx;

  coeff1[indi] = dydx_i;
  coeff2[indi] = (slope - dydx_i) / dx - t;
  coeff3[indi] = t/dx;
}




__device__
void prep_splines(int i, int length, int mode_i, int numModes, int interp_i, int ninterps,  double *b, double *ud, double *diag, double *ld, double *x, double *y){
  double dx1, dx2, d, slope1, slope2;
  int ind1x, ind2x, ind3x, ind1y, ind2y, ind3y;
  if (i == length - 1){

     ind1x = (length - 2) * ninterps + interp_i;
     ind2x = (length - 3) * ninterps + interp_i;
     ind3x = (length - 1) * ninterps + interp_i;

     ind1y = ((length - 2) * numModes + mode_i) * ninterps + interp_i;
     ind2y = ((length - 3) * numModes + mode_i) * ninterps + interp_i;
     ind3y = ((length - 1) * numModes + mode_i) * ninterps + interp_i;


  } else if (i == 0){

      ind1x = 1 * ninterps + interp_i;
      ind2x = 0 * ninterps + interp_i;
      ind3x = 2 * ninterps + interp_i;

      ind1y = (1 * numModes + mode_i) * ninterps + interp_i;
      ind2y = (0 * numModes + mode_i) * ninterps + interp_i;
      ind3y = (2 * numModes + mode_i) * ninterps + interp_i;


  } else{

      ind1x = (i) * ninterps + interp_i;
      ind2x = (i-1) * ninterps + interp_i;
      ind3x = (i+1) * ninterps + interp_i;

      ind1y = ((i) * numModes + mode_i) * ninterps + interp_i;
      ind2y = ((i-1) * numModes + mode_i) * ninterps + interp_i;
      ind3y = ((i+1) * numModes + mode_i) * ninterps + interp_i;
  }

    dx1 = x[ind1x] - x[ind2x];
    dx2 = x[ind3x] - x[ind1x];

    //amp
    slope1 = (y[ind1y] - y[ind2y])/dx1;
    slope2 = (y[ind3y] - y[ind1y])/dx2;

    b[ind1y] = 3.0* (dx2*slope1 + dx1*slope2);
    diag[ind1y] = 2*(dx1 + dx2);
    ud[ind1y] = dx1;
    ld[ind1y] = dx2;
}



CUDA_KERNEL
void fill_B(double *x_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
                      int ninterps, int length, int numModes){


    int start1 = blockIdx.x*blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = blockDim.x*gridDim.x;

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1)
        {

       for (int mode_i = 0; mode_i < numModes; mode_i += 1)
       {
           for (int i = start2;
                i < end2;
                i += diff2)
                {
                    prep_splines(i, length, mode_i, numModes, interp_i, ninterps,  B, upper_diag, diag, lower_diag, x_arr, y_all);

                }
       }

    }
}



CUDA_KERNEL
void set_spline_constants(double *x_arr, double *interp_array, double *B,
                      int ninterps, int length, int numModes){

    double dx;
    InterpContainer mode_vals;

    int start1 = blockIdx.x*blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = blockDim.x*gridDim.x;

    int npts = ninterps * length * numModes;

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

             for (int mode_i = 0; mode_i < numModes; mode_i += 1)
             {
                 for (int i = start2;
                      i < end2;
                      i += diff2)
                      {
                          dx = x_arr[i + 1] - x_arr[i];

                          int lead_ind = interp_i*length;
                          fill_coefficients(i, length, mode_i, numModes, interp_i, ninterps, B, dx,
                                            &interp_array[0 * npts],
                                            &interp_array[1 * npts],
                                            &interp_array[2 * npts],
                                            &interp_array[3 * npts]);

                      }
             }
}



void fit_wrap(int m, int n, double *a, double *b, double *c, double *d_in){

    #ifdef __CUDACC__
    size_t bufferSizeInBytes;

    cusparseHandle_t handle;
    void *pBuffer;

    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL( cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
    gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

    CUSPARSE_CALL(cusparseDgtsv2StridedBatch(handle,
                                              m,
                                              a, // dl
                                              b, //diag
                                              c, // du
                                              d_in,
                                              n,
                                              m,
                                              pBuffer));

  CUSPARSE_CALL(cusparseDestroy(handle));
  gpuErrchk(cudaFree(pBuffer));

  #else

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
for (int j = 0;
     j < n;
     j += 1){
       //fit_constants_serial(m, n, w, a, b, c, d_in, x_in, j);
       int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j*m + 1], &b[j*m], &c[j*m], &d_in[j*m], m);
       //if (info != m) printf("lapack info check: %d\n", info);

   }

  #endif

}
*/
