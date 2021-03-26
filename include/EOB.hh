#ifndef __EOB_HH__
#define __EOB_HH__

#include "global.h"

void compute_hlms_wrap(cmplx* hlms, double* r_arr, double* phi_arr, double* pr_arr, double* L_arr,
                  double* m1_arr, double* m2_arr, double* chi1_arr, double* chi2_arr,
                  int* num_steps, int num_steps_max, int* ell_arr_in, int* mm_arr_in, int num_modes, int num_bin_all);

#endif // __EOB_HH__
