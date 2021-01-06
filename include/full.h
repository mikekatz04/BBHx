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


void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths);

void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels);

void direct_sum(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end);

void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll);


#endif // __FULL_H__
