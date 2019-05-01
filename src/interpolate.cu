#include "manager.hh"
#include "stdio.h"
#include <assert.h>
#include <cusparse_v2.h>
#include "globalPhenomHM.h"

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)

#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)
using namespace std;


void host_fill_B_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            if (i == f_length - 1){
                B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i] - mode_vals[mode_i].amp[i-1]);
                B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i] - mode_vals[mode_i].phase[i-1]);
            } else if (i == 0){
                B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[1] - mode_vals[mode_i].amp[0]);
                B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0]);
            } else{
                B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i+1] - mode_vals[mode_i].amp[i-1]);
                B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i-1]);
            }
        }
    }
}

void host_fill_B_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            if (i == f_length - 1){
                B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i] - mode_vals[mode_i].phaseRdelay[i-1]);
                B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i] - mode_vals[mode_i].transferL1_re[i-1]);
                B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i] - mode_vals[mode_i].transferL1_im[i-1]);
                B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i] - mode_vals[mode_i].transferL2_re[i-1]);
                B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i] - mode_vals[mode_i].transferL2_im[i-1]);
                B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i] - mode_vals[mode_i].transferL3_re[i-1]);
                B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i] - mode_vals[mode_i].transferL3_im[i-1]);

            } else if (i == 0){
                B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[1] - mode_vals[mode_i].phaseRdelay[0]);
                B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[1] - mode_vals[mode_i].transferL1_re[0]);
                B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[1] - mode_vals[mode_i].transferL1_im[0]);
                B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[1] - mode_vals[mode_i].transferL2_re[0]);
                B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[1] - mode_vals[mode_i].transferL2_im[0]);
                B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[1] - mode_vals[mode_i].transferL3_re[0]);
                B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[1] - mode_vals[mode_i].transferL3_im[0]);
            } else{
                B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i+1] - mode_vals[mode_i].phaseRdelay[i-1]);
                B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i+1] - mode_vals[mode_i].transferL1_re[i-1]);
                B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i+1] - mode_vals[mode_i].transferL1_im[i-1]);
                B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i+1] - mode_vals[mode_i].transferL2_re[i-1]);
                B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i+1] - mode_vals[mode_i].transferL2_im[i-1]);
                B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i+1] - mode_vals[mode_i].transferL3_re[i-1]);
                B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i+1] - mode_vals[mode_i].transferL3_im[i-1]);
            }
        }
    }
}

__global__
void fill_B_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;

            if (i == f_length - 1){
                B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i] - mode_vals[mode_i].phaseRdelay[i-1]);
                B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i] - mode_vals[mode_i].transferL1_re[i-1]);
                B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i] - mode_vals[mode_i].transferL1_im[i-1]);
                B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i] - mode_vals[mode_i].transferL2_re[i-1]);
                B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i] - mode_vals[mode_i].transferL2_im[i-1]);
                B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i] - mode_vals[mode_i].transferL3_re[i-1]);
                B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i] - mode_vals[mode_i].transferL3_im[i-1]);

            } else if (i == 0){
                B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[1] - mode_vals[mode_i].phaseRdelay[0]);
                B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[1] - mode_vals[mode_i].transferL1_re[0]);
                B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[1] - mode_vals[mode_i].transferL1_im[0]);
                B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[1] - mode_vals[mode_i].transferL2_re[0]);
                B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[1] - mode_vals[mode_i].transferL2_im[0]);
                B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[1] - mode_vals[mode_i].transferL3_re[0]);
                B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[1] - mode_vals[mode_i].transferL3_im[0]);
            } else{
                B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i+1] - mode_vals[mode_i].phaseRdelay[i-1]);
                B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i+1] - mode_vals[mode_i].transferL1_re[i-1]);
                B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i+1] - mode_vals[mode_i].transferL1_im[i-1]);
                B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i+1] - mode_vals[mode_i].transferL2_re[i-1]);
                B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i+1] - mode_vals[mode_i].transferL2_im[i-1]);
                B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i+1] - mode_vals[mode_i].transferL3_re[i-1]);
                B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i+1] - mode_vals[mode_i].transferL3_im[i-1]);
            }
}

__global__ void fill_B_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;
    if (i == f_length - 1){
        B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i] - mode_vals[mode_i].amp[i-1]);
        B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i] - mode_vals[mode_i].phase[i-1]);
    } else if (i == 0){
        B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[1] - mode_vals[mode_i].amp[0]);
        B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0]);
    } else{
        B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i+1] - mode_vals[mode_i].amp[i-1]);
        B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i-1]);
    }
}

void host_set_spline_constants_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    double D_i, D_ip1, y_i, y_ip1;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            D_i = B[mode_i*f_length + i];
            D_ip1 = B[mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].amp[i];
            y_ip1 = mode_vals[mode_i].amp[i+1];
            mode_vals[mode_i].amp_coeff_1[i] = D_i;
            mode_vals[mode_i].amp_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].amp_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].phase[i];
            y_ip1 = mode_vals[mode_i].phase[i+1];
            mode_vals[mode_i].phase_coeff_1[i] = D_i;
            mode_vals[mode_i].phase_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].phase_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
        }
    }
}

void host_set_spline_constants_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    double D_i, D_ip1, y_i, y_ip1;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            D_i = B[(0*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(0*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].phaseRdelay[i];
            y_ip1 = mode_vals[mode_i].phaseRdelay[i+1];
            mode_vals[mode_i].phaseRdelay_coeff_1[i] = D_i;
            mode_vals[mode_i].phaseRdelay_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].phaseRdelay_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(1*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(1*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL1_re[i];
            y_ip1 = mode_vals[mode_i].transferL1_re[i+1];
            mode_vals[mode_i].transferL1_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL1_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL1_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(2*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(2*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL1_im[i];
            y_ip1 = mode_vals[mode_i].transferL1_im[i+1];
            mode_vals[mode_i].transferL1_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL1_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL1_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(3*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(3*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL2_re[i];
            y_ip1 = mode_vals[mode_i].transferL2_re[i+1];
            mode_vals[mode_i].transferL2_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL2_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL2_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(4*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(4*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL2_im[i];
            y_ip1 = mode_vals[mode_i].transferL2_im[i+1];
            mode_vals[mode_i].transferL2_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL2_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL2_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(5*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(5*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_re[i];
            y_ip1 = mode_vals[mode_i].transferL3_re[i+1];
            mode_vals[mode_i].transferL3_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(6*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(6*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_im[i];
            y_ip1 = mode_vals[mode_i].transferL3_im[i+1];
            mode_vals[mode_i].transferL3_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
        }
    }
}

__global__
void set_spline_constants_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    double D_i, D_ip1, y_i, y_ip1;
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;

            D_i = B[(0*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(0*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].phaseRdelay[i];
            y_ip1 = mode_vals[mode_i].phaseRdelay[i+1];
            mode_vals[mode_i].phaseRdelay_coeff_1[i] = D_i;
            mode_vals[mode_i].phaseRdelay_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].phaseRdelay_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(1*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(1*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL1_re[i];
            y_ip1 = mode_vals[mode_i].transferL1_re[i+1];
            mode_vals[mode_i].transferL1_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL1_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL1_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(2*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(2*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL1_im[i];
            y_ip1 = mode_vals[mode_i].transferL1_im[i+1];
            mode_vals[mode_i].transferL1_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL1_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL1_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(3*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(3*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL2_re[i];
            y_ip1 = mode_vals[mode_i].transferL2_re[i+1];
            mode_vals[mode_i].transferL2_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL2_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL2_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(4*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(4*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL2_im[i];
            y_ip1 = mode_vals[mode_i].transferL2_im[i+1];
            mode_vals[mode_i].transferL2_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL2_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL2_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(5*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(5*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_re[i];
            y_ip1 = mode_vals[mode_i].transferL3_re[i+1];
            mode_vals[mode_i].transferL3_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = B[(6*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = B[(6*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_im[i];
            y_ip1 = mode_vals[mode_i].transferL3_im[i+1];
            mode_vals[mode_i].transferL3_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
}

__global__ void set_spline_constants_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length-1) return;
    if (mode_i >= num_modes) return;
    double D_i, D_ip1, y_i, y_ip1;

    D_i = B[mode_i*f_length + i];
    D_ip1 = B[mode_i*f_length + i + 1];
    y_i = mode_vals[mode_i].amp[i];
    y_ip1 = mode_vals[mode_i].amp[i+1];
    mode_vals[mode_i].amp_coeff_1[i] = D_i;
    mode_vals[mode_i].amp_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
    mode_vals[mode_i].amp_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

    D_i = B[(num_modes*f_length) + mode_i*f_length + i];
    D_ip1 = B[(num_modes*f_length) + mode_i*f_length + i + 1];
    y_i = mode_vals[mode_i].phase[i];
    y_ip1 = mode_vals[mode_i].phase[i+1];
    mode_vals[mode_i].phase_coeff_1[i] = D_i;
    mode_vals[mode_i].phase_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
    mode_vals[mode_i].phase_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
}


void host_interpolate(cmplx *X_out, cmplx *Y_out, cmplx *Z_out, ModeContainer* old_mode_vals, int num_modes, double f_min, double df, double d_log10f, double *old_freqs, int old_length, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv){

    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cmplx fastPart;
    cmplx I(0.0, 1.0);
    double f_min_limit = old_freqs[0];
    double f_max_limit = old_freqs[old_length-1];
    int old_ind_below;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<length; i++){
            f = f_min + df * i;
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit)){
                X_out[mode_i*length + i] = 0.0+0.0*I;
                Y_out[mode_i*length + i] = 0.0+0.0*I;
                Z_out[mode_i*length + i] = 0.0+0.0*I;
                continue;
            }
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;
            // interp amplitude
            coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
            if (coeff_0 < 1e-50){
                X_out[mode_i*length + i] = cmplx(0.0, 0.0);
                Y_out[mode_i*length + i] = cmplx(0.0, 0.0);
                Z_out[mode_i*length + i] = cmplx(0.0, 0.0);
                continue;
            }
            coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            if (amp < 1e-40){
                X_out[mode_i*length + i] = 0.0+I*0.0;
                Y_out[mode_i*length + i] = 0.0+I*0.0;
                Z_out[mode_i*length + i] = 0.0+I*0.0;
                continue;
            }

            // interp phase
            coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

            phase  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].phaseRdelay[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phaseRdelay_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phaseRdelay_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phaseRdelay_coeff_3[old_ind_below];

            phaseRdelay  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            phasetimeshift = 2.*PI*(t0+tRef)*f;
            fastPart = amp * exp(I*(phase + phaseRdelay + phasetimeshift));

            // X
            coeff_0 = old_mode_vals[mode_i].transferL1_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_re_coeff_3[old_ind_below];

            transferL1_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL1_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_im_coeff_3[old_ind_below];

            transferL1_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            X_out[mode_i*length + i] = ((transferL1_re+I*transferL1_im) * fastPart * X_ASD_inv[i]);

            // Y
            coeff_0 = old_mode_vals[mode_i].transferL2_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_re_coeff_3[old_ind_below];

            transferL2_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL2_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_im_coeff_3[old_ind_below];

            transferL2_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            Y_out[mode_i*length + i] = ((transferL2_re+I*transferL2_im) * fastPart * Y_ASD_inv[i]);

            // Z
            coeff_0 = old_mode_vals[mode_i].transferL3_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_re_coeff_3[old_ind_below];

            transferL3_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL3_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_im_coeff_3[old_ind_below];

            transferL3_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            Z_out[mode_i*length + i] = ((transferL3_re+I*transferL3_im) * fastPart * Z_ASD_inv[i]);
        }
    }
}

__device__
cuDoubleComplex d_complex_exp (cuDoubleComplex arg)
{
   cuDoubleComplex res;
   double s, c;
   double e = exp(arg.x);
   sincos(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}


__global__
void interpolate(cuDoubleComplex *X_out, cuDoubleComplex *Y_out, cuDoubleComplex *Z_out, ModeContainer* old_mode_vals, int num_modes, double f_min, double df, double d_log10f, double *old_freqs, int old_length, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    double f_min_limit = old_freqs[0];
    double f_max_limit = old_freqs[old_length-1];
    cuDoubleComplex fastPart;
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    int old_ind_below;
            f = f_min + df * i;
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit)){
                X_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                Y_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                Z_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                return;
            }
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;
            // interp amplitude
            coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
            /*if (coeff_0 < 1e-50){
                X_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                Y_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                Z_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                continue;
            }*/
            coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            if (amp < 1e-40){
                X_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                Y_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                Z_out[mode_i*length + i] = make_cuDoubleComplex(0.0, 0.0);
                return;
            }
            // interp phase
            coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

            phase  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].phaseRdelay[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phaseRdelay_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phaseRdelay_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phaseRdelay_coeff_3[old_ind_below];

            phaseRdelay  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            phasetimeshift = 2.*PI*(t0+tRef)*f;
            fastPart = cuCmul(make_cuDoubleComplex(amp,0.0), d_complex_exp(cuCmul(I,make_cuDoubleComplex(phase + phaseRdelay + phasetimeshift, 0.0))));

            // X
            coeff_0 = old_mode_vals[mode_i].transferL1_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_re_coeff_3[old_ind_below];

            transferL1_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL1_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_im_coeff_3[old_ind_below];

            transferL1_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            X_out[mode_i*length + i] = cuCmul(cuCmul(make_cuDoubleComplex(transferL1_re, transferL1_im), fastPart), make_cuDoubleComplex(X_ASD_inv[i], 0.0)); //TODO may be faster to load as complex number with 0.0 for imaginary part

            // Y
            coeff_0 = old_mode_vals[mode_i].transferL2_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_re_coeff_3[old_ind_below];

            transferL2_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL2_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_im_coeff_3[old_ind_below];

            transferL2_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            Y_out[mode_i*length + i] = cuCmul(cuCmul(make_cuDoubleComplex(transferL2_re, transferL2_im), fastPart), make_cuDoubleComplex(Y_ASD_inv[i], 0.0));

            // Z
            coeff_0 = old_mode_vals[mode_i].transferL3_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_re_coeff_3[old_ind_below];

            transferL3_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL3_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_im_coeff_3[old_ind_below];

            transferL3_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            Z_out[mode_i*length + i] = cuCmul(cuCmul(make_cuDoubleComplex(transferL3_re, transferL3_im), fastPart), make_cuDoubleComplex(Z_ASD_inv[i], 0.0));
}

Interpolate::Interpolate(){
    int pass = 0;
}
void Interpolate::prep(double *B, int m_, int n_, int to_gpu_){
    m = m_;
    n = n_;
    to_gpu = to_gpu_;

    dl = new double[m];
    d = new double[m];
    du = new double[m];

    dl[0] = 0.0;
    d[0] = 2.0;
    du[0] = 1.0;
    d[m-1] = 2.0;
    du[m-1] = 0.0;
    dl[m-1] = 1.0;
    int i;
    for (i=1; i<m-1; i++){
        dl[i] = 1.0;
        du[i] = 1.0;
        d[i] = 4.0;
    }

    err = cudaMalloc(&d_dl, m*sizeof(double));
    assert(err == 0);
    err = cudaMalloc(&d_d, m*sizeof(double));
    assert(err == 0);
    err = cudaMalloc(&d_du, m*sizeof(double));
    assert(err == 0);
    err = cudaMemcpy(d_dl, dl, m*sizeof(double), cudaMemcpyHostToDevice);
    assert(err == 0);
    err = cudaMemcpy(d_d, d, m*sizeof(double), cudaMemcpyHostToDevice);
    assert(err == 0);
    err = cudaMemcpy(d_du, du, m*sizeof(double), cudaMemcpyHostToDevice);
    assert(err == 0);

    Interpolate::gpu_fit_constants(B);
    cudaFree(d_dl);
    cudaFree(d_du);
    cudaFree(d_d);
    delete[] d;
    delete[] dl;
    delete[] du;
    //dx_old = x_old[1] - x_old[0];
}

__host__ void Interpolate::gpu_fit_constants(double *B){
    /*double *h_B;
    int f_length= 20;
    int num_modes = 6;
    h_B = new double[2*f_length*num_modes];
    cudaMemcpy(h_B, B, 2*f_length*num_modes*sizeof(double), cudaMemcpyDeviceToHost);
    for (int i=0; i<2*f_length*num_modes; i++) printf("%e\n", h_B[i]);
    h_B = new double[2*f_length*num_modes];*/
    CUSPARSE_CALL( cusparseCreate(&handle) );
    cusparseStatus_t status = cusparseDgtsv(handle, m, n, d_dl, d_d, d_du, B, m);
    if (status !=  CUSPARSE_STATUS_SUCCESS) assert(0);
    cusparseDestroy(handle);
}

__host__ void Interpolate::fit_constants(double *B){
    int i;
    double *w = new double[m];
    double *D = new double[m];
    for (i=2; i<m; i++){
        //printf("%d\n", i);
        w[i] = dl[i]/d[i-1];
        d[i] = d[i] - w[i]*du[i-1];
        B[i] = B[i] - w[i]*B[i-1];
        //printf("%lf, %lf, %lf\n", w[i], d[i], b[i]);
    }

    D[m-1] = B[m-1]/d[m-1];
    for (i=(m-2); i>=0; i--){
        D[i] = (B[i] - du[i]*D[i+1])/d[i];
    }

    for (int i=0; i<m; i++) B[i] = D[i];
    delete[] D;
    delete[] w;
    /*for (i=0;i<N-1; i++){
        coeff_1[i] = D[i];
        coeff_2[i] = 3.0*(y_old[i+1] - y_old[i]) - 2.0*D[i] - D[i+1];
        coeff_3[i] = 2.0*(y_old[i] - y_old[i+1]) + D[i] + D[i+1];
    }*/
}

__host__ void Interpolate::transferToDevice(){
    cudaMemcpy(dev_coeff_1, coeff_1, (N-1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_coeff_2, coeff_2, (N-1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_coeff_3, coeff_3, (N-1)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x_old, x_old, (N)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y_old, y_old, (N)*sizeof(double), cudaMemcpyHostToDevice);
}

__device__ double Interpolate::call(double x_new){
    int index;
    if ((x_new > dev_x_old[0]) && (x_new < dev_x_old[N-1])){
        index = (int)floor(x_new/dx_old);
    } else if (x_new <= dev_x_old[0]){
        index = 0;
    } else {
        index = N-1;
    }
    double x = x_new - dev_x_old[index];
    double x2 = x*x;
    double x3 = x2*x;
    double y_new = dev_y_old[index] + dev_coeff_1[index]*x + dev_coeff_2[index]*x2 + dev_coeff_3[index]*x3;
    return y_new;
}

__host__ double Interpolate::cpu_call(double x_new){
    int index;
    if ((x_new > x_old[0]) && (x_new < x_old[N-1])){
        index = (int)floor(x_new/dx_old);
    } else if (x_new <= x_old[0]){
        index = 0;
    } else {
        index = N-1;
    }

    double x = x_new - x_old[index];
    double x2 = x*x;
    double x3 = x2*x;
    double y_new = y_old[index] + coeff_1[index]*x + coeff_2[index]*x2 + coeff_3[index]*x3;
    return y_new;
}

__host__ Interpolate::~Interpolate(){
}
