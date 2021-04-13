import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "EOB.hh":
    ctypedef void* cmplx 'cmplx'
    void compute_hlms_wrap(cmplx* hlms, double* r_arr, double* phi_arr, double* pr_arr, double* L_arr,
                  double* m1_arr, double* m2_arr, double* chi1_arr, double* chi2_arr,
                  int* num_steps, int num_steps_max, int* ell_arr_in, int* mm_arr_in, int num_modes, int num_bin_all);

    void root_find_all_wrap(double* xOut, double* x0In, double*argsIn, double* additionalArgsIn, int max_iter, double err, int numBinAll, int n, int num_args, int num_add_args);
    void root_find_scalar_all_wrap(double* pr_res, double* start_bounds, double*argsIn, double* additionalArgsIn, int max_iter, double err, int numBinAll, int num_args, int num_add_args);



@pointer_adjust
def compute_hlms(hlms, r_arr, phi_arr, pr_arr, L_arr,
              m1_arr, m2_arr, chi1_arr, chi2_arr,
              num_steps, num_steps_max, ell_arr_in, mm_arr_in, num_modes, num_bin_all):

    cdef size_t hlms_in = hlms
    cdef size_t r_arr_in = r_arr
    cdef size_t phi_arr_in = phi_arr
    cdef size_t pr_arr_in = pr_arr
    cdef size_t L_arr_in = L_arr
    cdef size_t m1_arr_in = m1_arr
    cdef size_t m2_arr_in = m2_arr
    cdef size_t chi1_arr_in = chi1_arr
    cdef size_t chi2_arr_in = chi2_arr
    cdef size_t num_steps_in = num_steps
    cdef size_t ell_arr_in_in = ell_arr_in
    cdef size_t mm_arr_in_in = mm_arr_in

    compute_hlms_wrap(<cmplx*> hlms_in, <double*> r_arr_in, <double*> phi_arr_in, <double*> pr_arr_in, <double*> L_arr_in,
                  <double*> m1_arr_in, <double*> m2_arr_in, <double*> chi1_arr_in, <double*> chi2_arr_in,
                  <int*> num_steps_in, num_steps_max, <int*> ell_arr_in_in, <int*> mm_arr_in_in, num_modes, num_bin_all)


@pointer_adjust
def root_find_all(xOut, x0In, argsIn, additionalArgsIn, max_iter, err, numBinAll, n, num_args, num_add_args):

    cdef size_t xOut_in = xOut
    cdef size_t x0In_in = x0In
    cdef size_t argsIn_in = argsIn
    cdef size_t additionalArgsIn_in = additionalArgsIn

    root_find_all_wrap(<double*> xOut_in, <double*> x0In_in, <double*> argsIn_in, <double*> additionalArgsIn_in, max_iter, err, numBinAll, n, num_args, num_add_args)

@pointer_adjust
def root_find_scalar_all(pr_res, start_bounds, argsIn, additionalArgsIn, max_iter, err, numBinAll, num_args, num_add_args):

    cdef size_t pr_res_in = pr_res
    cdef size_t start_bounds_in = start_bounds
    cdef size_t argsIn_in = argsIn
    cdef size_t additionalArgsIn_in = additionalArgsIn

    root_find_scalar_all_wrap(<double*> pr_res_in, <double*> start_bounds_in, <double*> argsIn_in, <double*> additionalArgsIn_in, max_iter, err, numBinAll, num_args, num_add_args)
