#include "manager.hh"
#include "stdio.h"
#include <assert.h>
#include <cusparse_v2.h>

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
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
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
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
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
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
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
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
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


void host_interpolate(std::complex<double> *X_out, std::complex<double> *Y_out, std::complex<double> *Z_out, ModeContainer* old_mode_vals, int num_modes, double f_min, double df, double d_log10f, double *old_freqs, int length, double tc, double tShift){

    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    std::complex<double> fastPart;
    std::complex<double> I(0.0, 1.0);
    int old_ind_below;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<length; i++){
            f = f_min + df * i;
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;
            // interp amplitude
            coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
            if (coeff_0 < 1e-50){
                X_out[mode_i*length + i] = std::complex<double>(0.0, 0.0);
                Y_out[mode_i*length + i] = std::complex<double>(0.0, 0.0);
                Z_out[mode_i*length + i] = std::complex<double>(0.0, 0.0);
                continue;
            }
            coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

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
            phasetimeshift = 2.*PI*(tc+tShift)*f;
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

            X_out[mode_i*length + i] = ((transferL1_re+I*transferL1_im) * fastPart);

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

            Y_out[mode_i*length + i] = ((transferL2_re+I*transferL2_im) * fastPart);

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

            Z_out[mode_i*length + i] = ((transferL3_re+I*transferL3_im) * fastPart);
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
void interpolate(cuDoubleComplex *X_out, cuDoubleComplex *Y_out, cuDoubleComplex *Z_out, ModeContainer* old_mode_vals, int num_modes, double f_min, double df, double d_log10f, double *old_freqs, int length, double tc, double tShift){
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= length) return;
    if (mode_i >= num_modes) return;
    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cuDoubleComplex fastPart;
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    int old_ind_below;

            f = f_min + df * i;
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
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
            phasetimeshift = 2.*PI*(tc+tShift)*f;
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

            //X_out[mode_i*length + i] = cuCmul(make_cuDoubleComplex(transferL1_re, transferL1_im), fastPart);

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

            //Y_out[mode_i*length + i] = cuCmul(make_cuDoubleComplex(transferL2_re, transferL2_im), fastPart);

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

            //Z_out[mode_i*length + i] = cuCmul(make_cuDoubleComplex(transferL3_re, transferL3_im), fastPart);
}

/*__global__ void interpolate2(cuDoubleComplex *hI_out,ModeContainer* old_mode_vals, int ind_min, int ind_max, int num_modes, double f_min, double df, int *old_inds, double *old_freqs, int length){
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.x;
    if (i + ind_min > ind_max) return;
    if (mode_i >= num_modes) return;
    int new_index = i + ind_min;
    int old_ind_below = old_inds[i];
    double f = f_min + df * new_index;
    double x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
    double x2 = x*x;
    double x3 = x*x2;
    double coeff_0, coeff_1, coeff_2, coeff_3;
    // interp amplitude
    coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
    coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
    coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
    coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

    double amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

    // interp phase
    coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
    coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
    coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
    coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

    double phase = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

    hI_out[mode_i*length + new_index] = make_cuDoubleComplex(amp*cos(phase), -1.0*amp*sin(phase));
}*/

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
    if (to_gpu == 1){
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
    }

    if (to_gpu == 1){
        Interpolate::gpu_fit_constants(B);
    }
    else Interpolate::fit_constants(B);
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
    delete D;
    delete w;
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
    delete dl;
    delete d;
    delete du;

    if (to_gpu == 1){
        cudaError_t err;
        err = cudaFree(d_dl);
        assert(err == 0);
        err = cudaFree(d_d);
        assert(err == 0);
        err = cudaFree(d_du);
        assert(err == 0);
        cusparseDestroy(handle);
    }

}

__global__ void run_interp(double *x_new, double *y_new, int num, Interpolate interp){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    y_new[i] = interp.call(x_new[i]);
}

__global__ void wave_interpolate(double *f_new, double *amp_new, double *phase_new, int num_modes, int length, Interpolate *interp_all){
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= length) return;
    int mode_i = blockIdx.x;
    if (mode_i >= num_modes) return;
    //Interpolate amp_interp = interp_all[mode_i];
    //Interpolate phase_interp = interp_all[num_modes + mode_i];
    amp_new[num_modes*length + i] = interp_all[mode_i].call(f_new[i]);
    phase_new[num_modes*length + i] = interp_all[num_modes + mode_i].call(f_new[i]);
}

/*
int main(){
    cudaError_t err;
    int num = 100;
    double *x = new double[num];
    double *y = new double[num];

    for (int i=0; i<num; i++){
        x[i] = (double) i+1.0;
        y[i] = x[i] * x[i] * x[i];
    }

    Interpolate interp;
    interp.prep(x, y, num);
    int new_num = 200;
    double *x_new = new double[new_num];
    double *y_new = new double[new_num];

    double dx = (x[num-1] - x[0])/(new_num + 1);
    for (int i=0; i<new_num; i++){
        x_new[i] = (i+1)*dx;
        y_new[i] = interp.cpu_call(x_new[i]);
        //y_new[i] = 0.0;
        printf("%lf, %lf\n", x_new[i], y_new[i]);
    }

    double *d_x_new, *d_y_new, *y_check;
    err = cudaMalloc(&d_x_new, new_num*sizeof(double));
    assert(err == 0);
    err = cudaMalloc(&d_y_new, new_num*sizeof(double));
    assert(err == 0);
    y_check = new double[new_num];

    err = cudaMemcpy(d_x_new, x_new, new_num*sizeof(double), cudaMemcpyHostToDevice);
    assert(err == 0);
    interp.transferToDevice();
    int NUM_THREADS = 256;
    int num_blocks = (new_num + NUM_THREADS -1)/NUM_THREADS;
    run_interp<<<num_blocks, NUM_THREADS>>>(d_x_new, d_y_new, new_num, interp);
    cudaDeviceSynchronize();

    err = cudaMemcpy(y_check, d_y_new, new_num*sizeof(double), cudaMemcpyDeviceToHost);
    assert(err == 0);
    for (int i=0; i<new_num; i++){
        //y_new[i] = 0.0;
        if (y_check[i] != y_new[i]) printf("%lf, %lf\n", y_new[i], y_check[i]);
    }

    //interp.Interpolate;
    err = cudaFree(d_x_new);
    assert(err == 0);
    err = cudaFree(d_y_new);
    assert(err == 0);
    delete x;
    delete y;
    delete x_new;
    delete y_new;
    delete y_check;

    printf("check\n");
    return(0);
}*/
