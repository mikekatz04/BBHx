import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Interpolate.hh":
    void interpolate(double* freqs, double* propArrays,
                     double* B, double* upper_diag, double* diag, double* lower_diag,
                     int length, int numInterpParams, int numModes, int numBinAll);
    void interpolate_TD(double* t_arr, double* propArrays,
                  double* B, double* upper_diag, double* diag, double* lower_diag,
                  int* lengths, int numBinAll, int nsubs);

@pointer_adjust
def interpolate_wrap(freqs, propArrays,
                     B, upper_diag, diag, lower_diag,
                     length, numInterpParams, numModes, numBinAll):

    cdef size_t freqs_in = freqs
    cdef size_t propArrays_in = propArrays
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag

    interpolate(<double*>freqs_in, <double*>propArrays_in,
              <double*>B_in, <double*>upper_diag_in, <double*>diag_in, <double*>lower_diag_in,
              length, numInterpParams, numModes, numBinAll)

@pointer_adjust
def interpolate_TD_wrap(t_arr, propArrays,
              B, upper_diag, diag, lower_diag,
              lengths, numBinAll, nsubs):

    cdef size_t t_arr_in = t_arr
    cdef size_t propArrays_in = propArrays
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag
    cdef size_t lengths_in = lengths

    interpolate_TD(<double*> t_arr_in, <double*> propArrays_in,
                  <double*> B_in, <double*> upper_diag_in, <double*> diag_in, <double*> lower_diag_in,
                  <int*> lengths_in, numBinAll, nsubs)
