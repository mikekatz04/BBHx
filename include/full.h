#ifndef __FULL_H__
#define __FULL_H__

#include <stdlib.h>
#include <complex>
#include "cuda_complex.hpp"
#include "stdio.h"

#include "global.h"


/**
 * @brief Solar mass, kg
 * @details
 * MSUN_SI = LAL_GMSUN_SI / LAL_G_SI
 */
//#define MSUN_SI 1.988546954961461467461011951140572744e30

/**
 * @brief Geometrized solar mass, s
 * @details
 * MTSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI * LAL_C_SI)
 */
//#define MTSUN_SI 4.925491025543575903411922162094833998e-6

/**
 * @brief Geometrized solar mass, m
 * @details
 * MRSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI)
 */
//#define MRSUN_SI 1.476625061404649406193430731479084713e3

/* CONSTANTS */



void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels);

void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll);


#endif // __FULL_H__
