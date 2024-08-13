#ifndef __RESPONSE_HH__
#define __RESPONSE_HH__

#include "global.h"
#include "Detector.hpp"

typedef struct tagd_Gslr_holder
{
    cmplx G21;
    cmplx G12;
    cmplx G23;
    cmplx G32;
    cmplx G31;
    cmplx G13;
} d_Gslr_holder;

typedef struct tagd_transferL_holder
{
    cmplx transferL1;
    cmplx transferL2;
    cmplx transferL3;
    double phaseRdelay;
} d_transferL_holder;

void LISA_response(
    double *response_out,
    int *ells_in,
    int *mms_in,
    double *freqs,   /**< Frequency points at which to evaluate the waveform (Hz) */
    double *phi_ref, /**< reference orbital phase (rad) */
    double *inc,
    double *lam,
    double *beta,
    double *psi,
    int TDItag, bool rescaled, bool tdi2, int order_fresnel_stencil,
    int numModes,
    int length,
    int numBinAll,
    int includesAmps,
    Orbits *orbits);

#endif // __RESPONSE_HH__
