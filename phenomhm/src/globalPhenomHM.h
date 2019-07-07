/*  This code was edited by Michael Katz. The information in this file is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited and added to
 *  the code for his purposes and removed dependencies on the LAL libraries.
 *  This code is distrbuted under the same GNU license it originally came with.

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


#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdlib.h>
#include <complex>


#define PI        3.141592653589793238462643383279502884
#define TWOPI     6.283185307179586476925286766559005768
#define PI_2      1.570796326794896619231321691639751442
#define PI_4      0.785398163397448309615660845819875721
#define MRSUN_SI  1.476625061404649406193430731479084713e+3
#define MTSUN_SI 4.925491025543575903411922162094833998e-6
#define MSUN_SI 1.988546954961461467461011951140572744e+30

#define GAMMA     0.577215664901532860606512090082402431
#define PC_SI 3.085677581491367278913937957796471611e16 /**< Parsec, m */

#define YRSID_SI 31558149.763545600

#define F0 3.168753578687779e-08
#define Omega0 1.9909865927683788e-07

#define ua 149597870700.
#define R_SI 149597870700.
#define AU_SI 149597870700.
#define aorbit 149597870700.

#define clight 299792458.0
#define sqrt3 1.7320508075688772
#define invsqrt3 0.5773502691896258
#define invsqrt6 0.4082482904638631
#define sqrt2 1.4142135623730951
#define L_SI 2.5e9
#define eorbit 0.004824185218078991
#define C_SI 299792458.0

#define NMODES_MAX 6

/**
 * Highest ell multipole PhenomHM models + 1.
 * Used to set array sizes
 */
#define L_MAX_PLUS_1 5

/* Activates amplitude part of the model */
#define AmpFlagTrue 1
#define AmpFlagFalse 0

/**
 * Dimensionless frequency (Mf) at which the inspiral amplitude
 * switches to the intermediate amplitude
 */
#define AMP_fJoin_INS 0.014

/**
 * Dimensionless frequency (Mf) at which the inspiral phase
 * switches to the intermediate phase
 */
#define PHI_fJoin_INS 0.018

typedef std::complex<double> cmplx;

typedef struct tagRealVector {
    double* data;
    size_t length;
} RealVector;

/*
* Structure for passing around PN phasing coefficients.
* For use with the TaylorF2 waveform.
*/
#define PN_PHASING_SERIES_MAX_ORDER 12
typedef struct tagPNPhasingSeries
{
   double v[PN_PHASING_SERIES_MAX_ORDER+1];
   double vlogv[PN_PHASING_SERIES_MAX_ORDER+1];
   double vlogvsq[PN_PHASING_SERIES_MAX_ORDER+1];
}
PNPhasingSeries;

typedef struct tagIMRPhenomDAmplitudeCoefficients {
  double eta;         // symmetric mass-ratio
  double chi1, chi2;  // dimensionless aligned spins, convention m1 >= m2.
  double q;           // asymmetric mass-ratio (q>=1)
  double chi;         // PN reduced spin parameter
  double fRD;         // ringdown frequency
  double fDM;         // imaginary part of the ringdown frequency (damping time)

  double fmaxCalc;    // frequency at which the mrerger-ringdown amplitude is maximum

  // Phenomenological inspiral amplitude coefficients
  double rho1;
  double rho2;
  double rho3;

  // Phenomenological intermediate amplitude coefficients
  double delta0;
  double delta1;
  double delta2;
  double delta3;
  double delta4;

  // Phenomenological merger-ringdown amplitude coefficients
  double gamma1;
  double gamma2;
  double gamma3;

  // Coefficients for collocation method. Used in intermediate amplitude model
  double f1, f2, f3;
  double v1, v2, v3;
  double d1, d2;

  // Transition frequencies for amplitude
  // We don't *have* to store them, but it may be clearer.
  double fInsJoin;    // Ins = Inspiral
  double fMRDJoin;    // MRD = Merger-Ringdown
}
IMRPhenomDAmplitudeCoefficients;

typedef struct tagIMRPhenomDPhaseCoefficients {
  double eta;         // symmetric mass-ratio
  double etaInv;      // 1/eta
  double eta2;        // eta*eta
  double Seta;        // sqrt(1.0 - 4.0*eta);
  double chi1, chi2;  // dimensionless aligned spins, convention m1 >= m2.
  double q;           // asymmetric mass-ratio (q>=1)
  double chi;         // PN reduced spin parameter
  double fRD;         // ringdown frequency
  double fDM;         // imaginary part of the ringdown frequency (damping time)

  // Phenomenological inspiral phase coefficients
  double sigma1;
  double sigma2;
  double sigma3;
  double sigma4;
  double sigma5;

  // Phenomenological intermediate phase coefficients
  double beta1;
  double beta2;
  double beta3;

  // Phenomenological merger-ringdown phase coefficients
  double alpha1;
  double alpha2;
  double alpha3;
  double alpha4;
  double alpha5;

  // C1 phase connection coefficients
  double C1Int;
  double C2Int;
  double C1MRD;
  double C2MRD;

  // Transition frequencies for phase
  double fInsJoin;    // Ins = Inspiral
  double fMRDJoin;    // MRD = Merger-Ringdown
}
IMRPhenomDPhaseCoefficients;

typedef struct tagUsefulPowers
{
    double sixth;
    double third;
    double two_thirds;
    double four_thirds;
    double five_thirds;
	double two;
    double seven_thirds;
    double eight_thirds;
} UsefulPowers;

typedef struct tagAmpInsPrefactors
{
	double two_thirds;
	double one;
	double four_thirds;
	double five_thirds;
	double two;
	double seven_thirds;
	double eight_thirds;
	double three;

	double amp0;
} AmpInsPrefactors;

typedef struct tagPhiInsPrefactors
{
	double initial_phasing;
	double third;
	double third_with_logv;
	double two_thirds;
	double one;
	double four_thirds;
	double five_thirds;
	double two;
	double logv;
	double minus_third;
	double minus_two_thirds;
	double minus_one;
	double minus_five_thirds;
} PhiInsPrefactors;


typedef struct tagPhenDAmpAndPhasePreComp
{
  PNPhasingSeries pn;
  IMRPhenomDPhaseCoefficients pPhi;
  PhiInsPrefactors phi_prefactors;
  IMRPhenomDAmplitudeCoefficients pAmp;
  AmpInsPrefactors amp_prefactors;
} PhenDAmpAndPhasePreComp;

/**
 * Structure storing pre-determined quantities
 * complying to the conventions of the PhenomHM model.
 * convensions such as m1>=m2
 */
typedef struct tagPhenomHMStorage
{
    double m1;    /**< mass of larger body in solar masses */
    double m2;    /**< mass of lighter body in solar masses */
    double m1_SI; /**< mass of larger body in kg */
    double m2_SI; /**< mass of lighter body in kg */
    double Mtot;  /**< total mass in solar masses */
    double eta;   /**< symmetric mass-ratio */
    double chi1z; /**< dimensionless aligned component spin of larger body */
    double chi2z; /**< dimensionless aligned component spin of lighter body */
    RealVector *freqs;
    double deltaF;
    double f_min;
    double f_max;
    double f_ref;
    double Mf_ref; /**< reference frequnecy in geometric units */
    double phiRef;
    unsigned int freq_is_uniform; /**< If = 1 then assume uniform spaced, If = 0 then assume arbitrarily spaced. */
    size_t npts;           /**< number of points in waveform array */
    size_t nmodes;           /**< number of modes */
    size_t ind_min;        /**< start index containing non-zero values */
    size_t ind_max;        /**< end index containing non-zero values */
    double finmass;
    double finspin;
    double Mf_RD_22;
    double Mf_DM_22;
    double PhenomHMfring[L_MAX_PLUS_1][L_MAX_PLUS_1];
    double PhenomHMfdamp[L_MAX_PLUS_1][L_MAX_PLUS_1];
    double Rholm[L_MAX_PLUS_1][L_MAX_PLUS_1]; /**< ratio of (2,2) mode to (l,m) mode ringdown frequency */
    double Taulm[L_MAX_PLUS_1][L_MAX_PLUS_1]; /**< ratio of (l,m) mode to (2,2) mode damping time */
} PhenomHMStorage;

/**
  * Structure holding Higher Mode Phase pre-computations
  */
typedef struct tagHMPhasePreComp
{
    double ai;
    double bi;
    double am;
    double bm;
    double ar;
    double br;
    double fi;
    double fr;
    double PhDBconst;
    double PhDCconst;
    double PhDBAterm;
} HMPhasePreComp;

// ModeContainer for this version of PhenomHM for gpu and cpu
typedef struct tagModeContainer{
    unsigned int l;
    unsigned int m;
    int num_modes;
    int length;
    int max_length;
    int to_gpu;
    int to_interp;

    double *freq_amp_phase;
    double d_log10f_amp_phase;
    double *amp;
    double *phase;
    double *time_freq_corr;
    double *amp_coeff_1;
    double *amp_coeff_2;
    double *amp_coeff_3;
    double *phase_coeff_1;
    double *phase_coeff_2;
    double *phase_coeff_3;

    double *time_freq_coeff_1;
    double *time_freq_coeff_2;
    double *time_freq_coeff_3;

    double *freq_response;
    double d_log10f_response;

    double *phaseRdelay;
    double *phaseRdelay_coeff_1;
    double *phaseRdelay_coeff_2;
    double *phaseRdelay_coeff_3;

    double *transferL1_re;
    double *transferL1_im;
    double *transferL1_re_coeff_1;
    double *transferL1_re_coeff_2;
    double *transferL1_re_coeff_3;
    double *transferL1_im_coeff_1;
    double *transferL1_im_coeff_2;
    double *transferL1_im_coeff_3;

    double *transferL2_re;
    double *transferL2_im;
    double *transferL2_re_coeff_1;
    double *transferL2_re_coeff_2;
    double *transferL2_re_coeff_3;
    double *transferL2_im_coeff_1;
    double *transferL2_im_coeff_2;
    double *transferL2_im_coeff_3;

    double *transferL3_re;
    double *transferL3_im;
    double *transferL3_re_coeff_1;
    double *transferL3_re_coeff_2;
    double *transferL3_re_coeff_3;
    double *transferL3_im_coeff_1;
    double *transferL3_im_coeff_2;
    double *transferL3_im_coeff_3;

} ModeContainer;

ModeContainer * cpu_create_modes(int num_modes, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp);
void cpu_destroy_modes(ModeContainer * mode_vals);

#endif
