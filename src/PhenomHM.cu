#include "PhenomHM.hh"
#include "global.h"
#include "constants.h"

#define NUM_THREADS_PHENOMHM 256
#define MAX_MODES 6
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
double pow_2_of(double number)
{
	return (number*number);
}

/**
 * calc cube of number without floating point 'pow'
 */
 CUDA_CALLABLE_MEMBER
double pow_3_of(double number)
{
	return (number*number*number);
}

/**
 * calc fourth power of number without floating point 'pow'
 */
CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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
    pfa->v6 += pfaN * (PI * (3760.L*SL + 1490.L*dSigmaL)/3.L + pn_ss3);
    pfa->v5 += pfaN * (-1.L * pn_gamma);
    pfa->vlogv5 += pfaN * (-3.L * pn_gamma);
    pfa->v4 += pfaN * (-10.L * pn_sigma);
    pfa->v3 += pfaN * (188.L*SL/3.L + 25.L*dSigmaL);
}




// From LALSimInspiralTaylorF2.c
/** \brief Returns structure containing TaylorF2 phasing coefficients for given
 *  physical parameters.
 */
 CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
double chiPN(double eta, double chi1, double chi2) {
  // Convention m1 >= m2 and chi1 is the spin on m1
  double delta = sqrt(1.0 - 4.0*eta);
  double chi_s = (chi1 + chi2) / 2.0;
  double chi_a = (chi1 - chi2) / 2.0;
  return chi_s * (1.0 - eta*76.0/113.0) + delta*chi_a;
}


CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
double amp0Func(double eta) {
  return (sqrt(2.0/3.0)*sqrt(eta))/pow(PI, 1./6.);
}


///////////////////////////// Amplitude: Inspiral functions /////////////////////////

// Phenom coefficients rho1, ..., rho3 from direct fit
// AmpInsDFFitCoeffChiPNFunc[eta, chiPN]

/**
 * rho_1 phenom coefficient. See corresponding row in Table 5 arXiv:1508.07253
 */
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
double AmpInsAnsatz(double Mf, UsefulPowers * powers_of_Mf, AmpInsPrefactors * prefactors) {
  double Mf2 = powers_of_Mf->two;
  double Mf3 = Mf*Mf2;

  return 1 + powers_of_Mf->two_thirds * prefactors->two_thirds
			+ Mf * prefactors->one + powers_of_Mf->four_thirds * prefactors->four_thirds
			+ powers_of_Mf->five_thirds * prefactors->five_thirds + Mf2 * prefactors->two
			+ powers_of_Mf->seven_thirds * prefactors->seven_thirds + powers_of_Mf->eight_thirds * prefactors->eight_thirds
			+ Mf3 * prefactors->three;
}

CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
double DPhiIntAnsatz(double Mf, IMRPhenomDPhaseCoefficients *p) {
  return (p->beta1 + p->beta3/pow_4_of(Mf) + p->beta2/Mf) / p->eta;
}

/**
 * temporary instance of DPhiIntAnsatz used when computing
 * coefficients to make the phase C(1) continuous between regions.
 */
 CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
double PhiInsAnsatzInt(double Mf, UsefulPowers *powers_of_Mf, PhiInsPrefactors *prefactors, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn)
{
	//assert(0 != pn);

  // Assemble PN phasing series
  const double v = powers_of_Mf->third * pow(PI, 1./3.);
  const double logv = log(v);


  double phasing = prefactors->initial_phasing;

  //if (Mf < 7.38824e-04) printf("PhenomHM 1: %e %e %e \n", Mf, p->eta, phasing);

  phasing += prefactors->two_thirds	* powers_of_Mf->two_thirds;
  phasing += prefactors->third * powers_of_Mf->third;
  phasing += prefactors->third_with_logv * logv * powers_of_Mf->third;

  //if (Mf < 7.38824e-04) printf("PhenomHM 2: %e %e %e \n", Mf, p->eta, phasing);

  phasing += prefactors->logv * logv;
  phasing += prefactors->minus_third / powers_of_Mf->third;
  phasing += prefactors->minus_two_thirds / powers_of_Mf->two_thirds;

   //if (Mf < 7.38824e-04) printf("PhenomHM 3: %e %e %e %e %e %e  \n", Mf, p->eta, phasing, prefactors->logv * logv, prefactors->minus_third / powers_of_Mf->third, prefactors->minus_two_thirds / powers_of_Mf->two_thirds);

  phasing += prefactors->minus_one / Mf;
  phasing += prefactors->minus_five_thirds / powers_of_Mf->five_thirds; // * v^0

  //if (Mf < 7.38824e-04) printf("PhenomHM 4: %e %e %e \n", Mf, p->eta, phasing);
  // Now add higher order terms that were calibrated for PhenomD
  phasing += ( prefactors->one * Mf + prefactors->four_thirds * powers_of_Mf->four_thirds
			   + prefactors->five_thirds * powers_of_Mf->five_thirds
			   + prefactors->two * powers_of_Mf->two
			 ) / p->eta;

  //if (Mf < 7.38824e-04) printf("PhenomHM 5: %e %e %e \n", Mf, p->eta, phasing);
  return phasing;
}

CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
double IMRPhenDPhaseDerivative(double f, IMRPhenomDPhaseCoefficients *p, PNPhasingSeries *pn, double Rholm, double Taulm)
{
 // Defined in VIII. Full IMR Waveforms arXiv:1508.07253
 // The inspiral, intermendiate and merger-ringdown phase parts

 // split the calculation to just 1 of 3 possible mutually exclusive ranges
 if (!StepFunc_boolean(f, p->fInsJoin))	// Inspiral range
 {
     double DPhiIns_val = DPhiInsAnsatzInt(f, p, pn);
     return DPhiIns_val;
 }

 if (StepFunc_boolean(f, p->fMRDJoin))	// MRD range
 {
     double DPhiMRD_val = DPhiMRD(f, p, Rholm, Taulm) + p->C2MRD;
     return DPhiMRD_val;
 }

 //	Intermediate range
 double DPhiInt_val = DPhiIntAnsatz(f, p) + p->C2Int;;
 return DPhiInt_val;
}

/**
 * Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
 * (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
 * was not available when PhenomD was tuned.
 */
CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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
 CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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

CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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


//////////////////////// Final spin, final mass, fring, fdamp ////////////////////////

// Final Spin and Radiated Energy formulas described in 1508.07250

/**
 * Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
 * s defined around Equation 3.6.
 */
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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
CUDA_CALLABLE_MEMBER
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


CUDA_KERNEL
void get_phenomhm_ringdown_frequencies(
    double *fringdown,
    double *fdamp,
    double *m1,
    double *m2,
    double *chi1z,
    double *chi2z,
    int *ells_in,
    int *mms_in,
    int numModes,
    int numBinAll
)
{
    CUDA_SHARED int ells[MAX_MODES];
    CUDA_SHARED int mms[MAX_MODES];

    int start, increment;
    #ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int i = start; i < numModes; i += increment)
    {
        ells[i] = ells_in[i];
        mms[i] = mms_in[i];
    }

    CUDA_SYNC_THREADS;

    #ifdef __CUDACC__
    start = threadIdx.x + blockDim.x * blockIdx.x;
    increment = gridDim.x * blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int binNum = start; binNum < numBinAll; binNum += increment)
    {
        double finmass = IMRPhenomDFinalMass(m1[binNum], m2[binNum], chi1z[binNum], chi2z[binNum]);
        double finspin = IMRPhenomDFinalSpin(m1[binNum], m2[binNum], chi1z[binNum], chi2z[binNum]);

        for (int mode_i = 0; mode_i < numModes; mode_i += 1)
        {
            unsigned int ell = (unsigned int) ells[mode_i];
            int mm = mms[mode_i];

            int index = binNum * numModes + mode_i;
            IMRPhenomHMGetRingdownFrequency(
                &fringdown[index],
                &fdamp[index],
                ell,
                mm,
                finmass,
                finspin);
        }
    }
}

void get_phenomhm_ringdown_frequencies_wrap(
    double *fringdown,
    double *fdamp,
    double *m1,
    double *m2,
    double *chi1z,
    double *chi2z,
    int *ells_in,
    int *mm_in,
    int numModes,
    int numBinAll
)
{
    int nblocks = std::ceil((numBinAll + NUM_THREADS_PHENOMHM -1)/NUM_THREADS_PHENOMHM);
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);*/
        //printf("%d begin\n", jj);
    #ifdef __CUDACC__
    get_phenomhm_ringdown_frequencies<<<nblocks, NUM_THREADS_PHENOMHM>>>(
        fringdown,
        fdamp,
        m1,
        m2,
        chi1z,
        chi2z,
        ells_in,
        mm_in,
        numModes,
        numBinAll
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else
    get_phenomhm_ringdown_frequencies(
        fringdown,
        fdamp,
        m1,
        m2,
        chi1z,
        chi2z,
        ells_in,
        mm_in,
        numModes,
        numBinAll
    );
    #endif

}

// removed +/- 0.999 so 1001 not 1003
#define NUM_SPLINE_PTS 1001
CUDA_KERNEL
void get_phenomd_ringdown_frequencies(
    double *fringdown,
    double *fdamp,
    double *m1,
    double *m2,
    double *chi1z,
    double *chi2z,
    int numBinAll,
    double *y_rd_all,
    double *c1_rd_all,
    double *c2_rd_all,
    double *c3_rd_all,
    double *y_dm_all,
    double *c1_dm_all,
    double *c2_dm_all,
    double *c3_dm_all,
    double dspin
)
{

    // TODO: constant memory?
    int start, increment;

    #ifdef __CUDACC__
    start = threadIdx.x + blockDim.x * blockIdx.x;
    increment = gridDim.x * blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int binNum = start; binNum < numBinAll; binNum += increment)
    {
        double finmass = IMRPhenomDFinalMass(m1[binNum], m2[binNum], chi1z[binNum], chi2z[binNum]);
        double finspin = IMRPhenomDFinalSpin(m1[binNum], m2[binNum], chi1z[binNum], chi2z[binNum]);

        double lowest_spin = -1.0;
        int ind = int( (finspin - lowest_spin) / dspin ); // -1.0 min spin
        double x_spl = dspin * ind + lowest_spin;

        double y_rd = y_rd_all[ind];
        double c1_rd = c1_rd_all[ind];
        double c2_rd = c2_rd_all[ind];
        double c3_rd = c3_rd_all[ind];

        double y_dm = y_dm_all[ind];
        double c1_dm = c1_dm_all[ind];
        double c2_dm = c2_dm_all[ind];
        double c3_dm = c3_dm_all[ind];

        double x = finspin - x_spl;
        double x2 = x * x;
        double x3 = x2 * x;

        //printf("%e %e %d %e %e %e %e %e %e\n", finmass, finspin, ind, x_spl, , );
        double fring_temp = y_rd + c1_rd * x + c2_rd * x2 + c3_rd * x3;
        double fdamp_temp = y_dm + c1_dm * x + c2_dm * x2 + c3_dm * x3;

        fringdown[binNum] = fring_temp / finmass;
        fdamp[binNum] = fdamp_temp / finmass;

    }
}

void get_phenomd_ringdown_frequencies_wrap(
    double *fringdown,
    double *fdamp,
    double *m1,
    double *m2,
    double *chi1z,
    double *chi2z,
    int numBinAll,
    double *y_rd_all,
    double *c1_rd_all,
    double *c2_rd_all,
    double *c3_rd_all,
    double *y_dm_all,
    double *c1_dm_all,
    double *c2_dm_all,
    double *c3_dm_all,
    double dspin
)
{
    int nblocks = std::ceil((numBinAll + NUM_THREADS_PHENOMHM -1)/NUM_THREADS_PHENOMHM);
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);*/
        //printf("%d begin\n", jj);
    #ifdef __CUDACC__
    get_phenomd_ringdown_frequencies<<<nblocks, NUM_THREADS_PHENOMHM>>>(
        fringdown,
        fdamp,
        m1,
        m2,
        chi1z,
        chi2z,
        numBinAll,
        y_rd_all,
        c1_rd_all,
        c2_rd_all,
        c3_rd_all,
        y_dm_all,
        c1_dm_all,
        c2_dm_all,
        c3_dm_all,
        dspin
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else
    get_phenomd_ringdown_frequencies(
        fringdown,
        fdamp,
        m1,
        m2,
        chi1z,
        chi2z,
        numBinAll,
        y_rd_all,
        c1_rd_all,
        c2_rd_all,
        c3_rd_all,
        y_dm_all,
        c1_dm_all,
        c2_dm_all,
        c3_dm_all,
        dspin
    );
    #endif

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
    const double f_ref,
    const double phi_ref
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
    p->phi_ref = phi_ref;

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
        double Rholm = pHM->Rholm;
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

CUDA_CALLABLE_MEMBER
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


CUDA_CALLABLE_MEMBER
void get_phase(int i, double* phase, double* tf, double freq_geom, int ell, int mm, PhenomHMStorage* pHM, UsefulPowers powers_of_f, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double cshift[], double Rholm, double Taulm, double t0, double phi0)
{
        double Mf_wf, Mfr, tmpphaseC, phase_term1, phase_term2, tf_i;
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
          tf_i = IMRPhenDPhaseDerivative(Mf, &pDPreComp.pPhi, &pDPreComp.pn, Rholm, Taulm);
      }
      else if (!(Mf_wf > q.fr))
      { /* in mathematica -> IMRPhenDPhaseB */
          Mf = q.am * Mf_wf + q.bm;

          phase_i += IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.am - q.PhDBconst + q.PhDBAterm;
          tf_i = IMRPhenDPhaseDerivative(Mf, &pDPreComp.pPhi, &pDPreComp.pn, Rholm, Taulm);
      }
      else if ((Mf_wf > q.fr))
      { /* in mathematica -> IMRPhenDPhaseC */
          Mfr = q.am * q.fr + q.bm;
          tmpphaseC = IMRPhenomDPhase_OneFrequency(Mfr, pDPreComp, Rholm, Taulm) / q.am - q.PhDBconst + q.PhDBAterm;
          Mf = q.ar * Mf_wf + q.br;
          phase_i += IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, Rholm, Taulm) / q.ar - q.PhDCconst + tmpphaseC;
          tf_i = IMRPhenDPhaseDerivative(Mf, &pDPreComp.pPhi, &pDPreComp.pn, Rholm, Taulm);
      }

      Mf = freq_geom;
      phase_term1 = - t0 * (Mf - pHM->Mf_ref);
      phase_term2 = phase_i - (mm * phi0);

      *phase = (phase_term1 + phase_term2);
      *tf = (tf_i) / (2. * PI);
}


 /**
  * Michael Katz added this function.
  * internal function that filles amplitude and phase for a specific frequency and mode.
  */

 CUDA_CALLABLE_MEMBER
 double get_phase_phenomd(double Mf, PhenDAmpAndPhasePreComp pDPreComp, double t0, double phi0, double Mf_ref, double cshift[])
 {
     int mm = 2;
     double phase_i = cshift[mm];

     phase_i += IMRPhenomDPhase_OneFrequency(Mf, pDPreComp, 1.0, 1.0);

     double phase_term1 = - t0 * (Mf - Mf_ref);
     double phase_term2 = phase_i - (mm * phi0);

     return (phase_term1 + phase_term2);
 }

CUDA_CALLABLE_MEMBER
void calculate_modes_phenomd(int binNum, double* amps, double* phases, double* tf, double* freqs, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp pDPreComp, double amp0, double t0, double phi0, int length, int numBinAll, double M_tot_sec, double Mf_ref, double cshift[])
{
    double eps = 1e-9;

    int start, increment;
    #ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int i = start; i < length; i += increment)
    {

        double amp_i, phase_i, dphidf, phase_up, phase_down;
        double t_wave_frame, t_sampling_frame;
        int status_in_for;
        UsefulPowers powers_of_f;
        int retcode = 0;

        int freq_index = binNum * length + i;

        double freq = freqs[freq_index];
        double freq_geom = freq*M_tot_sec;

        status_in_for = init_useful_powers(&powers_of_f, freq_geom);

        amp_i = IMRPhenDAmplitude(freq_geom, pAmp, &powers_of_f, &amp_prefactors);

        phase_i = get_phase_phenomd(freq_geom, pDPreComp, t0, phi0, Mf_ref, cshift);

        amps[freq_index] = amp_i * amp0;
        phases[freq_index] = phase_i;

        dphidf = M_tot_sec * IMRPhenDPhaseDerivative(freq_geom, &pDPreComp.pPhi, &pDPreComp.pn, 1.0, 1.0) / (2 * PI);
        tf[freq_index] = dphidf - (t0 / (2. * PI) * M_tot_sec);

    }
}

 CUDA_CALLABLE_MEMBER
 void calculate_modes(int binNum, int mode_i, double* amps, double* phases, double* tf, double* freqs, int ell, int mm, PhenomHMStorage *pHM, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double amp0, double Rholm, double Taulm, double t0, double phi0, int length, int numBinAll, int numModes, double M_tot_sec, double cshift[])
 {
         double eps = 1e-9;

         int start, increment;
         #ifdef __CUDACC__
         start = threadIdx.x;
         increment = blockDim.x;
         #else
         start = 0;
         increment = 1;
         #pragma omp parallel for
         #endif
         for (int i = start; i < length; i += increment)
         {

             double amp_i, phase_i, tf_i, dphidf, phase_up, phase_down;
             double t_wave_frame, t_sampling_frame;
             int status_in_for;
             UsefulPowers powers_of_f;
             int retcode = 0;

             int mode_index = (binNum * numModes + mode_i) * length + i;
             int freq_index = binNum * length + i;

             double freq = freqs[freq_index];
             double freq_geom = freq*M_tot_sec;

             get_amp(&amp_i, freq_geom, ell, mm, pHM, powers_of_f, pAmp, amp_prefactors, amp0);

             get_phase(i, &phase_i, &tf_i, freq_geom, ell, mm, pHM, powers_of_f, pDPreComp, q, cshift, Rholm, Taulm, t0, phi0);

             amps[mode_index] = amp_i;

             phases[mode_index] = phase_i;

             tf[mode_index] = (M_tot_sec * tf_i) - (t0 / (2. * PI) * M_tot_sec);


         }

}




/**
 * Michael Katz added this function.
 * Main function for calculating PhenomHM in the form used by Michael Katz
 * This is setup to allow for pre-allocation of arrays. Therefore, all arrays
 * should be setup outside of this function.
 */
CUDA_CALLABLE_MEMBER
void IMRPhenomHMCore(
    int *ells,
    int *mms,
    double* amps,
    double* phases,
    double* tf,
    double* freqs,                      /**< GW frequecny list [Hz] */
    double m1_SI,                               /**< primary mass [kg] */
    double m2_SI,                               /**< secondary mass [kg] */
    double chi1z,                               /**< aligned spin of primary */
    double chi2z,                               /**< aligned spin of secondary */
    const double distance,                      /**< distance [m] */
    double f_ref,
    int length,                              /**< reference GW frequency */
    int numModes,
    int binNum,
    int numBinAll,
    double cshift[],
    double* Mf_RD_lm,
    double* Mf_DM_lm,
    int run_phenomd
)
{

    // TODO: run_phenomd int -> bool

    // set phi_ref to zero
    double phi_ref = 0.0;
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
        phi_ref
    );


    /* populate the ringdown frequency array */
    /* If you want to model a new mode then you have to add it here. */
    /* (l,m) = (2,2) */



    if (!run_phenomd)
    {
        IMRPhenomHMGetRingdownFrequency(
            &pHM->Mf_RD_22,
            &pHM->Mf_DM_22,
            2, 2,
            pHM->finmass, pHM->finspin);
    }
    else
    {
        pHM->Mf_RD_22 = Mf_RD_lm[0];
        pHM->Mf_DM_22 = Mf_DM_lm[0];
    }

    /* (l,m) = (2,2) */
    int ell, mm;
    ell = 2;
    mm = 2;
    pHM->Rho22 = 1.0;
    pHM->Tau22 = 1.0;

    double Mf_RD_22_in, Mf_DM_22_in;

    if (!run_phenomd)
    {
        Mf_RD_22_in = Mf_RD_lm[numModes];
        Mf_DM_22_in = Mf_DM_lm[numModes];
    }
    else
    {
        Mf_RD_22_in = Mf_RD_lm[0];
        Mf_DM_22_in = Mf_DM_lm[0];
    }

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
        Mf_RD_22_in,
        Mf_DM_22_in
    );

    // set f_ref to f_max

    //if (pHM->f_ref == 0.0){
        pHM->Mf_ref = pDPreComp22.pAmp.fmaxCalc;

        pHM->f_ref = PhenomUtilsMftoHz(pHM->Mf_ref, pHM->Mtot);
        //printf("%e, %e\n", pHM->f_ref, pHM->Mf_ref);
    //}

    /* compute the reference phase shift need to align the waveform so that
     the phase is equal to phi_ref at the reference frequency f_ref. */
    /* the phase shift is computed by evaluating the phase of the
    (l,m)=(2,2) mode.
    phi0 is the correction we need to add to each mode. */
    double phi_22_at_f_ref = IMRPhenomDPhase_OneFrequency(pHM->Mf_ref, pDPreComp22,  1.0, 1.0);

    // REMINDER: phi_ref is set to zero
    phi0 = 0.5 * (phi_22_at_f_ref + phi_ref);

    //t0 = IMRPhenomDComputet0(pHM->eta, pHM->chi1z, pHM->chi2z, pHM->finspin, &(pDPreComp22.pPhi), &(pDPreComp22.pAmp));
    t0 = IMRPhenDPhaseDerivative(pHM->Mf_ref, &pDPreComp22.pPhi, &pDPreComp22.pn, 1.0, 1.0);

    // setup PhenomD info. Sub here is due to preallocated struct

    retcode = 0;

    const double Mtot = (m1_SI + m2_SI) / MSUN_SI;

   /* Compute the amplitude pre-factor */
   // amp0 is passed into this function as a pointer.This is for compatibility with GPU.
   amp0 = PhenomUtilsFDamp0(Mtot, distance); // TODO check if this is right units
    double M_tot_sec = (pHM->m1 + pHM->m2)*MTSUN_SI;

    if (run_phenomd)
    {
        calculate_modes_phenomd(binNum, amps, phases, tf, freqs,  &(pDPreComp22.pAmp), pDPreComp22.amp_prefactors, pDPreComp22, amp0, t0, phi0, length, numBinAll, M_tot_sec, pHM->Mf_ref, cshift);
    }
    else
    {
        //HMPhasePreComp q;

        // prep q and pDPreComp for each mode in the loop below
        HMPhasePreComp qlm;
        PhenDAmpAndPhasePreComp pDPreComplm;

        double Rholm, Taulm;

        for (int mode_i=0; mode_i<numModes; mode_i++)
        {
            ell = ells[mode_i];
            mm = mms[mode_i];

            pHM->Mf_RD_lm = Mf_RD_lm[mode_i];
            pHM->Mf_DM_lm = Mf_DM_lm[mode_i];

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
                Mf_RD_22_in,
                Mf_DM_22_in);
                //pHM->Mf_RD_lm,
                //pHM->Mf_DM_lm);

            retcode = IMRPhenomHMPhasePreComp(&qlm, ell, mm, pHM, pDPreComplm);

            calculate_modes(binNum, mode_i, amps, phases, tf, freqs, ell, mm, pHM, &(pDPreComplm.pAmp), pDPreComplm.amp_prefactors, pDPreComplm, qlm, amp0, Rholm, Taulm, t0, phi0, length, numBinAll, numModes, M_tot_sec, cshift);

        }
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

 CUDA_KERNEL
 void IMRPhenomHM(
     double* amps, /**< [out] Frequency-domain waveform hx */
     double* phases,
     double* tf,
     int* ells_in,
     int* mms_in,
     double* freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
     double* m1_SI,                        /**< mass of companion 1 (kg) */
     double* m2_SI,                        /**< mass of companion 2 (kg) */
     double* chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
     double* chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
     double* distance,               /**< distance of source (m) */
     double* f_ref,                        /**< Reference frequency */
     int numModes,
     int length,
     int numBinAll,
     double* Mf_RD_lm_all,
     double* Mf_DM_lm_all,
     int run_phenomd
)
{

    /*
     * Phase shift due to leading order complex amplitude
     * [L.Blancet, arXiv:1310.1528 (Sec. 9.5)]
     * "Spherical hrmonic modes for numerical relativity"
     */
    /* List of phase shifts: the index is the azimuthal number m */
    CUDA_SHARED double cShift[7];

    CUDA_SHARED int ells[MAX_MODES];
    CUDA_SHARED int mms[MAX_MODES];

    #ifdef __CUDACC__
    CUDA_SHARED double Mf_RD_lm[MAX_MODES + 1];
    CUDA_SHARED double Mf_DM_lm[MAX_MODES + 1];
    #endif

    if THREAD_ZERO
    {
        cShift[0] = 0.0;
        cShift[1] = PI_2; /* i shift */
        cShift[2] = 0.0;
        cShift[3] = -PI_2; /* -i shift */
        cShift[4] = PI; /* 1 shift */
        cShift[5] = PI_2; /* -1 shift */
        cShift[6] = 0.0;
    }

    CUDA_SYNC_THREADS;

    int start, increment;
    #ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int i = start; i < numModes; i += increment)
    {
        ells[i] = ells_in[i];
        mms[i] = mms_in[i];
    }

    CUDA_SYNC_THREADS;

    #ifdef __CUDACC__
    start = blockIdx.x;
    increment = gridDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int binNum = start; binNum < numBinAll; binNum += increment)
    {

        int add = 0;
        if (!run_phenomd) add = 1;
        #ifdef __CUDACC__
        #else
        double Mf_RD_lm[MAX_MODES + 1];
        double Mf_DM_lm[MAX_MODES + 1];
        #endif

        int start2, increment2;
        #ifdef __CUDACC__
        start2 = threadIdx.x;
        increment2 = blockDim.x;
        #else
        start2 = 0;
        increment2 = 1;
        #pragma omp parallel for
        #endif
        for (int i = start2; i < numModes + add; i += increment2)
        {
            Mf_RD_lm[i] = Mf_RD_lm_all[binNum * (numModes + add) + i];
            Mf_DM_lm[i] = Mf_DM_lm_all[binNum * (numModes + add) + i];
        }
        CUDA_SYNC_THREADS;

        IMRPhenomHMCore(ells, mms, amps, phases, tf, freqs, m1_SI[binNum], m2_SI[binNum], chi1z[binNum], chi2z[binNum], distance[binNum], f_ref[binNum], length, numModes, binNum, numBinAll, cShift, Mf_RD_lm, Mf_DM_lm, run_phenomd);
    }
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
    double* f_ref,                        /**< Reference frequency */
    int numModes,
    int length,
    int numBinAll,
    double* Mf_RD_lm_all,
    double* Mf_DM_lm_all,
    int run_phenomd
)
{

    double* amps = &waveformOut[0];
    double* phases = &waveformOut[numBinAll * numModes * length];
    double* tf = &waveformOut[2 * numBinAll * numModes * length];

    //int nblocks = std::ceil((numBinAll + NUM_THREADS_PHENOMHM -1)/NUM_THREADS_PHENOMHM);
    int nblocks = numBinAll; //std::ceil((numBinAll + NUM_THREADS_PHENOMHM -1)/NUM_THREADS_PHENOMHM);
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);*/
        //printf("%d begin\n", jj);
    #ifdef __CUDACC__
    IMRPhenomHM<<<nblocks, NUM_THREADS_PHENOMHM>>>(
        amps, /**< [out] Frequency-domain waveform hx */
        phases,
        tf,
        ells_in,
        mms_in,
        freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
        m1_SI,                        /**< mass of companion 1 (kg) */
        m2_SI,                        /**< mass of companion 2 (kg) */
        chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
        chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
        distance,               /**< distance of source (m) */
        f_ref,                        /**< Reference frequency */
        numModes,
        length,
        numBinAll,
        Mf_RD_lm_all,
        Mf_DM_lm_all,
        run_phenomd
    );

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else
    IMRPhenomHM(
        amps, /**< [out] Frequency-domain waveform hx */
        phases,
        tf,
        ells_in,
        mms_in,
        freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
        m1_SI,                        /**< mass of companion 1 (kg) */
        m2_SI,                        /**< mass of companion 2 (kg) */
        chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
        chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
        distance,               /**< distance of source (m) */
        f_ref,                        /**< Reference frequency */
        numModes,
        length,
        numBinAll,
        Mf_RD_lm_all,
        Mf_DM_lm_all,
        run_phenomd
    );
    #endif

    /*
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%e\n", milliseconds);*/

}
