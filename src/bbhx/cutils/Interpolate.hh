#ifndef __INTERPOLATE_BBH_HH__
#define __INTERPOLATE_BBH_HH__

#include "gbt_global.h"

void interpolate(double* freqs, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll);

#endif // __INTERPOLATE_BBH_HH__
