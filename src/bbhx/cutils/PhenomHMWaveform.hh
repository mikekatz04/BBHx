#ifndef __PHENOMHM__
#define __PHENOMHM__


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



typedef struct tagPNPhasingSeries
{
   double v0;
   double v1;
   double v2;
   double v3;
   double v4;
   double v5;
   double v6;
   double v7;
   double v8;
   double v9;
   double v10;

   double vlogv0;
   double vlogv1;
   double vlogv2;
   double vlogv3;
   double vlogv4;
   double vlogv5;
   double vlogv6;
   double vlogv7;
   double vlogv8;
   double vlogv9;
   double vlogv10;

   double vlogvsq0;
   double vlogvsq1;
   double vlogvsq2;
   double vlogvsq3;
   double vlogvsq4;
   double vlogvsq5;
   double vlogvsq6;
   double vlogvsq7;
   double vlogvsq8;
   double vlogvsq9;
   double vlogvsq10;
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
    double deltaF;
    double f_min;
    double f_max;
    double f_ref;
    double Mf_ref; /**< reference frequnecy in geometric units */
    double phi_ref;
    unsigned int freq_is_uniform; /**< If = 1 then assume uniform spaced, If = 0 then assume arbitrarily spaced. */
    int npts;           /**< number of points in waveform array */
    int nmodes;           /**< number of modes */
    int ind_min;        /**< start index containing non-zero values */
    int ind_max;        /**< end index containing non-zero values */
    double finmass;
    double finspin;
    double Mf_RD_22;
    double Mf_DM_22;
    double Mf_RD_lm;
    double Mf_DM_lm;
    double Rholm; /**< ratio of (2,2) mode to (l,m) mode ringdown frequency */
    double Taulm; /**< ratio of (l,m) mode to (2,2) mode damping time */
    double Rho22;
    double Tau22;
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
);

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
);

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
);

#endif // __PHENOMHM__
