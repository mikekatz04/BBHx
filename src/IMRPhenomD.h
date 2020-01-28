/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown in `PhenomHM.cpp`. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.
 */

#ifndef _IMR_PHENOMD_H
#define _IMR_PHENOMD_H

#include "IMRPhenomD_internals.h"
#include "globalPhenomHM.h"

int IMRPhenomDGenerateFD(
    COMPLEX16FrequencySeries **htilde,  /**< [out] FD waveform */
    const double phi0,                  /**< Orbital phase at fRef (rad) */
    const double fRef_in,               /**< reference frequency (Hz) */
    const double deltaF,                /**< Sampling frequency (Hz) */
    const double m1_SI,                 /**< Mass of companion 1 (kg) */
    const double m2_SI,                 /**< Mass of companion 2 (kg) */
    const double chi1,                  /**< Aligned-spin parameter of companion 1 */
    const double chi2,                  /**< Aligned-spin parameter of companion 2 */
    const double f_min,                 /**< Starting GW frequency (Hz) */
    const double f_max,                 /**< End frequency; 0 defaults to Mf = \ref f_CUT */
    const double distance               /**< Distance of source (m) */
);

int IMRPhenomDGenerateh22FDAmpPhase(
    AmpPhaseFDWaveform** h22,           /**< [out] FD waveform */
    RealVector* freq,                  /**< Input: frequencies (Hz) on which to evaluate h22 FD - will be copied in the output AmpPhaseFDWaveform. Frequencies exceeding max freq covered by PhenomD will be given 0 amplitude and phase. */
    double* constants_main,
    double* constants_amp,
    double* constants_phase,
    const double phi0,                  /**< Orbital phase at fRef (rad) */
    const double fRef_in,               /**< reference frequency (Hz) */
    const double m1_SI,                 /**< Mass of companion 1 (kg) */
    const double m2_SI,                 /**< Mass of companion 2 (kg) */
    const double chi1,                  /**< Aligned-spin parameter of companion 1 */
    const double chi2,                  /**< Aligned-spin parameter of companion 2 */
    const double distance               /**< Distance of source (m) */
);

int IMRPhenomDSetupAmpAndPhaseCoefficients(
   PhenDAmpAndPhasePreComp *pDPreComp,
   double m1,
   double m2,
   double chi1z,
   double chi2z,
   const double Rholm,
   const double Taulm);

CUDA_CALLABLE_MEMBER
double IMRPhenomDPhase_OneFrequency(
       double Mf,
       PhenDAmpAndPhasePreComp pD,
       double Rholm,
       double Taulm);

double IMRPhenomDComputet0(
           double eta,           /**< symmetric mass-ratio */
           double chi1z,         /**< dimensionless aligned-spin of primary */
           double chi2z,         /**< dimensionless aligned-spin of secondary */
           double finspin       /**< final spin */
       );

double IMRPhenomDFinalSpin(
   const double m1_in,                 /**< mass of companion 1 [Msun] */
   const double m2_in,                 /**< mass of companion 2 [Msun] */
   const double chi1_in,               /**< aligned-spin of companion 1 */
   const double chi2_in               /**< aligned-spin of companion 2 */
);

double IMRPhenomDFinalMass(
    double m1,    /**< mass of primary in solar masses */
    double m2,    /**< mass of secondary in solar masses */
    double chi1z, /**< aligned-spin component on primary */
    double chi2z  /**< aligned-spin component on secondary */
);

int ins_IMRPhenomDSetupAmpAndPhaseCoefficients(
   PhenDAmpAndPhasePreComp *pDPreComp,
   double m1,
   double m2,
   double chi1z,
   double chi2z);

#endif /* _LALSIM_IMR_PHENOMD_H */
