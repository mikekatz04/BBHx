#ifndef __INTERPOLATE_HH__
#define __INTERPOLATE_HH__

void interpolate(double* freqs, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll);

#endif // __INTERPOLATE_HH__
