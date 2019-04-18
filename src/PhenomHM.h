#ifndef _IMR_PHENOMHM_H
#define _IMR_PHENOMHM_H


#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "globalPhenomHM.h"
#include "IMRPhenomD_internals.h"

/**
 * Dimensionless frequency (Mf) at which define the end of the waveform
 */
#define PHENOMHM_DEFAULT_MF_MAX 0.5

/**
 * eta is the symmetric mass-ratio.
 * This number corresponds to a mass-ratio of 20
 * The underlying PhenomD model was calibrated to mass-ratio 18
 * simulations. We choose mass-ratio 20 as a conservative
 * limit on where the model should be accurate.
 */
#define MAX_ALLOWED_ETA 0.045351

/**
 * Maximum number of (l,m) mode paris PhenomHM models.
 * Only count positive 'm' modes.
 * Used to set default mode array
 */
#define NMODES_MAX 6

/**
 * Highest ell multipole PhenomHM models + 1.
 * Used to set array sizes
 */
//#define L_MAX_PLUS_1 5

/* Activates amplitude part of the model */
//#define AmpFlagTrue 1
//#define AmpFlagFalse 0

//#define PI_2 1.5707963268

//LALDict *IMRPhenomHM_setup_mode_array(
//    LALDict *extraParams);

/**
 * useful powers in GW waveforms: 1/6, 1/3, 2/3, 4/3, 5/3, 2, 7/3, 8/3, -1, -1/6, -7/6, -1/3, -2/3, -5/3
 * calculated using only one invocation of 'pow', the rest are just multiplications and divisions
 */
typedef struct tagPhenomHMUsefulPowers
{
    double third;
    double two_thirds;
    double four_thirds;
    double five_thirds;
    double two;
    double seven_thirds;
    double eight_thirds;
    double inv;
    double m_seven_sixths;
    double m_third;
    double m_two_thirds;
    double m_five_thirds;
} PhenomHMUsefulPowers;

/**
  * Useful powers of Mf: 1/6, 1/3, 2/3, 4/3, 5/3, 2, 7/3, 8/3, -7/6, -5/6, -1/2, -1/6, 1/2
  * calculated using only one invocation of 'pow' and one of 'sqrt'.
  * The rest are just multiplications and divisions.  Also including Mf itself in here.
  */
typedef struct tagPhenomHMUsefulMfPowers
{
    double itself;
    double sixth;
    double third;
    double two_thirds;
    double four_thirds;
    double five_thirds;
    double two;
    double seven_thirds;
    double eight_thirds;
    double m_seven_sixths;
    double m_five_sixths;
    double m_sqrt;
    double m_sixth;
    double sqrt;
} PhenomHMUsefulMfPowers;

/**
 * must be called before the first usage of *p
 */
int PhenomHM_init_useful_mf_powers(PhenomHMUsefulMfPowers *p, double number);

/**
 * must be called before the first usage of *p
 */
int PhenomHM_init_useful_powers(PhenomHMUsefulPowers *p, double number);


/**
 * Structure storing pre-determined quantities
 * that describe the frequency array
 * and tells you over which indices will contain non-zero values.
 */
typedef struct tagPhenomHMFrequencyBoundsStorage
{
    double deltaF;
    double f_min;
    double f_max;
    double f_ref;
    unsigned int freq_is_uniform; /**< If = 1 then assume uniform spaced, If = 0 then assume arbitrarily spaced. */
    size_t npts;           /**< number of points in waveform array */
    size_t ind_min;        /**< start index containing non-zero values */
    size_t ind_max;        /**< end index containing non-zero values */
} PhenomHMFrequencyBoundsStorage;

int init_IMRPhenomHMGet_FrequencyBounds_storage(
    PhenomHMFrequencyBoundsStorage *p,
    RealVector *freqs,
    double Mtot,
    double deltaF,
    double f_ref_in);

unsigned int IMRPhenomHM_is_freq_uniform(
    RealVector *freqs,
    double deltaF);


static int init_PhenomHM_Storage(
    PhenomHMStorage *p,   /**< [out] PhenomHMStorage struct */
    const double m1_SI,    /**< mass of companion 1 (kg) */
    const double m2_SI,    /**< mass of companion 2 (kg) */
    const double chi1z,    /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
    const double chi2z,    /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
    RealVector *freqs, /**< Input frequency sequence (Hz) */
    const double deltaF,   /**< frequency spacing (Hz) */
    const double f_ref,    /**< reference GW frequency (hz) */
    const double phiRef    /**< orbital phase at f_ref */
);


double IMRPhenomHMTrd(
    double Mf,
    double Mf_RD_22,
    double Mf_RD_lm,
    const int AmpFlag,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM);

double IMRPhenomHMTi(
    double Mf,
    const int mm);

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
    PhenomHMStorage *pHM);

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
    double Br);

int IMRPhenomHMFreqDomainMapParams(
    double *a,
    double *b,
    double *fi,
    double *fr,
    double *f1,
    const double flm,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM,
    const int AmpFlag);

double IMRPhenomHMFreqDomainMap(
    double Mflm,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM,
    const int AmpFlag);

int IMRPhenomHMPhasePreComp(
    HMPhasePreComp *q,
    const int ell,
    const int mm,
    PhenomHMStorage *pHM);

double IMRPhenomHMOnePointFiveSpinPN(
    double fM,
    int l,
    int m,
    double M1,
    double M2,
    double X1z,
    double X2z);

int IMRPhenomHM(
    ModeContainer *mode_vals, /**< [out] Frequency-domain waveform hx */
    double *freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
    int f_length,
    double m1_SI,                        /**< mass of companion 1 (kg) */
    double m2_SI,                        /**< mass of companion 2 (kg) */
    double chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
    double chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
    const double distance,               /**< distance of source (m) */
    const double inclination,            /**< inclination of source (rad) */
    const double phiRef,                 /**< reference orbital phase (rad) */
    const double deltaF,                 /**< Sampling frequency (Hz). To use arbitrary frequency points set deltaF <= 0. */
    double f_ref,                        /**< Reference frequency */
    int num_modes,
    int to_gpu
);

int IMRPhenomHMCore(
    ModeContainer *mode_vals, /**< [out] Frequency domain hx GW strain */
    double *freqs_trans,                      /**< GW frequecny list [Hz] */
    int f_length,
    double m1_SI,                               /**< primary mass [kg] */
    double m2_SI,                               /**< secondary mass [kg] */
    double chi1z,                               /**< aligned spin of primary */
    double chi2z,                               /**< aligned spin of secondary */
    const double distance,                      /**< distance [m] */
    const double inclination,                   /**< inclination angle */
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
);

void host_calculate_all_modes(ModeContainer *mode_vals, PhenomHMStorage *pHM, double *freqs, double M_tot_sec, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp *pDPreComp_all, HMPhasePreComp *q_all, double amp0, int num_modes, double t0, double phi0);

void host_calculate_each_mode(int i, ModeContainer mode_val, unsigned int ell, unsigned int mm, PhenomHMStorage *pHM, double freq_geom, IMRPhenomDAmplitudeCoefficients *pAmp, AmpInsPrefactors amp_prefactors, PhenDAmpAndPhasePreComp pDPreComp, HMPhasePreComp q, double amp0, double Rholm, double Taulm, double t0, double phi0);

int IMRPhenomHMGetRingdownFrequency(
    double *fringdown,
    double *fdamp,
    unsigned int ell,
    int mm,
    double finalmass,
    double finalspin);

#endif /* _LALSIM_IMR_PHENOMHM_H */
