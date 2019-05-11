#include "c_manager.h"
#include "stdio.h"
#include <assert.h>
#include "globalPhenomHM.h"

using namespace std;

void host_fill_B_wave(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            if (i == f_length - 1){
                b_mat[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i] - mode_vals[mode_i].amp[i-1]);
                b_mat[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i] - mode_vals[mode_i].phase[i-1]);
            } else if (i == 0){
                b_mat[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[1] - mode_vals[mode_i].amp[0]);
                b_mat[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0]);
            } else{
                b_mat[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i+1] - mode_vals[mode_i].amp[i-1]);
                b_mat[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i-1]);
            }
        }
    }
}

void host_fill_B_response(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes){
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            if (i == f_length - 1){
                b_mat[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i] - mode_vals[mode_i].phaseRdelay[i-1]);
                b_mat[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i] - mode_vals[mode_i].transferL1_re[i-1]);
                b_mat[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i] - mode_vals[mode_i].transferL1_im[i-1]);
                b_mat[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i] - mode_vals[mode_i].transferL2_re[i-1]);
                b_mat[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i] - mode_vals[mode_i].transferL2_im[i-1]);
                b_mat[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i] - mode_vals[mode_i].transferL3_re[i-1]);
                b_mat[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i] - mode_vals[mode_i].transferL3_im[i-1]);
                b_mat[(7*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].time_freq_corr[i] - mode_vals[mode_i].time_freq_corr[i-1]);

            } else if (i == 0){
                b_mat[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[1] - mode_vals[mode_i].phaseRdelay[0]);
                b_mat[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[1] - mode_vals[mode_i].transferL1_re[0]);
                b_mat[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[1] - mode_vals[mode_i].transferL1_im[0]);
                b_mat[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[1] - mode_vals[mode_i].transferL2_re[0]);
                b_mat[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[1] - mode_vals[mode_i].transferL2_im[0]);
                b_mat[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[1] - mode_vals[mode_i].transferL3_re[0]);
                b_mat[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[1] - mode_vals[mode_i].transferL3_im[0]);
                b_mat[(7*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].time_freq_corr[1] - mode_vals[mode_i].time_freq_corr[0]);
            } else{
                b_mat[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i+1] - mode_vals[mode_i].phaseRdelay[i-1]);
                b_mat[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i+1] - mode_vals[mode_i].transferL1_re[i-1]);
                b_mat[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i+1] - mode_vals[mode_i].transferL1_im[i-1]);
                b_mat[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i+1] - mode_vals[mode_i].transferL2_re[i-1]);
                b_mat[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i+1] - mode_vals[mode_i].transferL2_im[i-1]);
                b_mat[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i+1] - mode_vals[mode_i].transferL3_re[i-1]);
                b_mat[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i+1] - mode_vals[mode_i].transferL3_im[i-1]);
                b_mat[(7*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].time_freq_corr[i+1] - mode_vals[mode_i].time_freq_corr[i-1]);
            }
        }
    }
}

void host_set_spline_constants_wave(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes){
    double D_i, D_ip1, y_i, y_ip1;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            D_i = b_mat[mode_i*f_length + i];
            D_ip1 = b_mat[mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].amp[i];
            y_ip1 = mode_vals[mode_i].amp[i+1];
            mode_vals[mode_i].amp_coeff_1[i] = D_i;
            mode_vals[mode_i].amp_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].amp_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].phase[i];
            y_ip1 = mode_vals[mode_i].phase[i+1];
            mode_vals[mode_i].phase_coeff_1[i] = D_i;
            mode_vals[mode_i].phase_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].phase_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
        }
    }
}

void host_set_spline_constants_response(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes){
    double D_i, D_ip1, y_i, y_ip1;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<f_length; i++){
            D_i = b_mat[(0*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(0*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].phaseRdelay[i];
            y_ip1 = mode_vals[mode_i].phaseRdelay[i+1];
            mode_vals[mode_i].phaseRdelay_coeff_1[i] = D_i;
            mode_vals[mode_i].phaseRdelay_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].phaseRdelay_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(1*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(1*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL1_re[i];
            y_ip1 = mode_vals[mode_i].transferL1_re[i+1];
            mode_vals[mode_i].transferL1_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL1_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL1_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(2*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(2*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL1_im[i];
            y_ip1 = mode_vals[mode_i].transferL1_im[i+1];
            mode_vals[mode_i].transferL1_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL1_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL1_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(3*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(3*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL2_re[i];
            y_ip1 = mode_vals[mode_i].transferL2_re[i+1];
            mode_vals[mode_i].transferL2_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL2_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL2_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(4*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(4*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL2_im[i];
            y_ip1 = mode_vals[mode_i].transferL2_im[i+1];
            mode_vals[mode_i].transferL2_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL2_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL2_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(5*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(5*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_re[i];
            y_ip1 = mode_vals[mode_i].transferL3_re[i+1];
            mode_vals[mode_i].transferL3_re_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(6*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(6*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_im[i];
            y_ip1 = mode_vals[mode_i].transferL3_im[i+1];
            mode_vals[mode_i].transferL3_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(6*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(6*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].transferL3_im[i];
            y_ip1 = mode_vals[mode_i].transferL3_im[i+1];
            mode_vals[mode_i].transferL3_im_coeff_1[i] = D_i;
            mode_vals[mode_i].transferL3_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].transferL3_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

            D_i = b_mat[(7*num_modes*f_length) + mode_i*f_length + i];
            D_ip1 = b_mat[(7*num_modes*f_length) + mode_i*f_length + i + 1];
            y_i = mode_vals[mode_i].time_freq_corr[i];
            y_ip1 = mode_vals[mode_i].time_freq_corr[i+1];
            mode_vals[mode_i].time_freq_coeff_1[i] = D_i;
            mode_vals[mode_i].time_freq_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
            mode_vals[mode_i].time_freq_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
        }
    }
}
void host_interpolate(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals, int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv){

    double f, time_start, x, x2, x3;
    double time_coeff_0, time_coeff_1, time_coeff_2, time_coeff_3;
    double amp_coeff_0, amp_coeff_1, amp_coeff_2, amp_coeff_3;
    double phase_coeff_0, phase_coeff_1, phase_coeff_2, phase_coeff_3;
    double phase_delay_coeff_0, phase_delay_coeff_1, phase_delay_coeff_2, phase_delay_coeff_3;
    double l1_re_coeff_0, l1_re_coeff_1, l1_re_coeff_2, l1_re_coeff_3;
    double l1_im_coeff_0, l1_im_coeff_1, l1_im_coeff_2, l1_im_coeff_3;
    double l2_re_coeff_0, l2_re_coeff_1, l2_re_coeff_2, l2_re_coeff_3;
    double l2_im_coeff_0, l2_im_coeff_1, l2_im_coeff_2, l2_im_coeff_3;
    double l3_re_coeff_0, l3_re_coeff_1, l3_re_coeff_2, l3_re_coeff_3;
    double l3_im_coeff_0, l3_im_coeff_1, l3_im_coeff_2, l3_im_coeff_3;

    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cmplx fastPart;
    cmplx I(0.0, 1.0);
    double f_min_limit = old_freqs[0];
    double f_max_limit = old_freqs[old_length-1];
    int old_ind_below;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        old_ind_below = -1;
        while (data_freqs[0] > old_freqs[old_ind_below+2]){
            old_ind_below++;
        }
        for (int i=0; i<length; i++){
            f = data_freqs[i];
            if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit)){
                channel1_out[mode_i*length + i] = 0.0+0.0*I;
                channel2_out[mode_i*length + i] = 0.0+0.0*I;
                channel3_out[mode_i*length + i] = 0.0+0.0*I;
                continue;
            }
            if (f >= old_freqs[old_ind_below+1]){
                old_ind_below += 1;
                if (old_ind_below == old_length -1){
                    channel1_out[mode_i*length + i] = 0.0+0.0*I;
                    channel2_out[mode_i*length + i] = 0.0+0.0*I;
                    channel3_out[mode_i*length + i] = 0.0+0.0*I;
                    continue;
                }
                time_coeff_0 = old_mode_vals[mode_i].time_freq_corr[old_ind_below];
                time_coeff_1 = old_mode_vals[mode_i].time_freq_coeff_1[old_ind_below];
                time_coeff_2 = old_mode_vals[mode_i].time_freq_coeff_2[old_ind_below];
                time_coeff_3 = old_mode_vals[mode_i].time_freq_coeff_3[old_ind_below];

                // interp amplitude
                amp_coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
                amp_coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
                amp_coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
                amp_coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

                phase_coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
                phase_coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
                phase_coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
                phase_coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

                phase_delay_coeff_0 = old_mode_vals[mode_i].phaseRdelay[old_ind_below];
                phase_delay_coeff_1 = old_mode_vals[mode_i].phaseRdelay_coeff_1[old_ind_below];
                phase_delay_coeff_2 = old_mode_vals[mode_i].phaseRdelay_coeff_2[old_ind_below];
                phase_delay_coeff_3 = old_mode_vals[mode_i].phaseRdelay_coeff_3[old_ind_below];

                // X
                l1_re_coeff_0 = old_mode_vals[mode_i].transferL1_re[old_ind_below];
                l1_re_coeff_1 = old_mode_vals[mode_i].transferL1_re_coeff_1[old_ind_below];
                l1_re_coeff_2 = old_mode_vals[mode_i].transferL1_re_coeff_2[old_ind_below];
                l1_re_coeff_3 = old_mode_vals[mode_i].transferL1_re_coeff_3[old_ind_below];

                l1_im_coeff_0 = old_mode_vals[mode_i].transferL1_im[old_ind_below];
                l1_im_coeff_1 = old_mode_vals[mode_i].transferL1_im_coeff_1[old_ind_below];
                l1_im_coeff_2 = old_mode_vals[mode_i].transferL1_im_coeff_2[old_ind_below];
                l1_im_coeff_3 = old_mode_vals[mode_i].transferL1_im_coeff_3[old_ind_below];

                // Y
                l2_re_coeff_0 = old_mode_vals[mode_i].transferL2_re[old_ind_below];
                l2_re_coeff_1 = old_mode_vals[mode_i].transferL2_re_coeff_1[old_ind_below];
                l2_re_coeff_2 = old_mode_vals[mode_i].transferL2_re_coeff_2[old_ind_below];
                l2_re_coeff_3 = old_mode_vals[mode_i].transferL2_re_coeff_3[old_ind_below];

                l2_im_coeff_0 = old_mode_vals[mode_i].transferL2_im[old_ind_below];
                l2_im_coeff_1 = old_mode_vals[mode_i].transferL2_im_coeff_1[old_ind_below];
                l2_im_coeff_2 = old_mode_vals[mode_i].transferL2_im_coeff_2[old_ind_below];
                l2_im_coeff_3 = old_mode_vals[mode_i].transferL2_im_coeff_3[old_ind_below];

                // Z
                l3_re_coeff_0 = old_mode_vals[mode_i].transferL3_re[old_ind_below];
                l3_re_coeff_1 = old_mode_vals[mode_i].transferL3_re_coeff_1[old_ind_below];
                l3_re_coeff_2 = old_mode_vals[mode_i].transferL3_re_coeff_2[old_ind_below];
                l3_re_coeff_3 = old_mode_vals[mode_i].transferL3_re_coeff_3[old_ind_below];

                l3_im_coeff_0 = old_mode_vals[mode_i].transferL3_im[old_ind_below];
                l3_im_coeff_1 = old_mode_vals[mode_i].transferL3_im_coeff_1[old_ind_below];
                l3_im_coeff_2 = old_mode_vals[mode_i].transferL3_im_coeff_2[old_ind_below];
                l3_im_coeff_3 = old_mode_vals[mode_i].transferL3_im_coeff_3[old_ind_below];
            };

            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;

            time_start = time_coeff_0 + (time_coeff_1*x) + (time_coeff_2*x2) + (time_coeff_3*x3);

            if (time_start <= 0.0) {
                channel1_out[mode_i*length + i] = 0.0+0.0*I;
                channel2_out[mode_i*length + i] = 0.0+0.0*I;
                channel3_out[mode_i*length + i] = 0.0+0.0*I;
                continue;
            }

            amp = amp_coeff_0 + (amp_coeff_1*x) + (amp_coeff_2*x2) + (amp_coeff_3*x3);
            if (amp < 1e-40){
                channel1_out[mode_i*length + i] = 0.0+I*0.0;
                channel2_out[mode_i*length + i] = 0.0+I*0.0;
                channel3_out[mode_i*length + i] = 0.0+I*0.0;
                continue;
            }

            // interp phase
            phase  = phase_coeff_0 + (phase_coeff_1*x) + (phase_coeff_2*x2) + (phase_coeff_3*x3);

            phaseRdelay  = phase_delay_coeff_0 + (phase_delay_coeff_1*x) + (phase_delay_coeff_2*x2) + (phase_delay_coeff_3*x3);
            phasetimeshift = 2.*PI*(t0+tRef)*f;
            fastPart = amp * exp(I*(phase + phaseRdelay + phasetimeshift));

            transferL1_re  = l1_re_coeff_0 + (l1_re_coeff_1*x) + (l1_re_coeff_2*x2) + (l1_re_coeff_3*x3);

            transferL1_im  = l1_im_coeff_0 + (l1_im_coeff_1*x) + (l1_im_coeff_2*x2) + (l1_im_coeff_3*x3);

            channel1_out[mode_i*length + i] = ((transferL1_re+I*transferL1_im) * fastPart * X_ASD_inv[i]);

            transferL2_re  = l2_re_coeff_0 + (l2_re_coeff_1*x) + (l2_re_coeff_2*x2) + (l2_re_coeff_3*x3);

            transferL2_im  = l2_im_coeff_0 + (l2_im_coeff_1*x) + (l2_im_coeff_2*x2) + (l2_im_coeff_3*x3);

            channel2_out[mode_i*length + i] = ((transferL2_re+I*transferL2_im) * fastPart * Y_ASD_inv[i]);

            transferL3_re  = l3_re_coeff_0 + (l3_re_coeff_1*x) + (l3_re_coeff_2*x2) + (l3_re_coeff_3*x3);

            transferL3_im  = l3_im_coeff_0 + (l3_im_coeff_1*x) + (l3_im_coeff_2*x2) + (l3_im_coeff_3*x3);

            channel3_out[mode_i*length + i] = ((transferL3_re+I*transferL3_im) * fastPart * Z_ASD_inv[i]);
        }
    }
}

/*
void host_interpolate(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals, int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv){

    double f, time_start, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cmplx fastPart;
    cmplx I(0.0, 1.0);
    double f_min_limit = old_freqs[0];
    double f_max_limit = old_freqs[old_length-1];
    int old_ind_below;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<length; i++){
            f = data_freqs[i];
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit)){
                channel1_out[mode_i*length + i] = 0.0+0.0*I;
                channel2_out[mode_i*length + i] = 0.0+0.0*I;
                channel3_out[mode_i*length + i] = 0.0+0.0*I;
                continue;
            }
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;

            // interp time frequency to remove less than 0.0
            coeff_0 = old_mode_vals[mode_i].time_freq_corr[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].time_freq_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].time_freq_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].time_freq_coeff_3[old_ind_below];

            time_start = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            if (i<5) printf("%e, %e, %e, %d\n", f, time_start, coeff_0, old_ind_below);
            if (time_start <= 0.0) {
                channel1_out[mode_i*length + i] = 0.0+0.0*I;
                channel2_out[mode_i*length + i] = 0.0+0.0*I;
                channel3_out[mode_i*length + i] = 0.0+0.0*I;
                continue;
            }

            // interp amplitude
            coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            if (amp < 1e-40){
                channel1_out[mode_i*length + i] = 0.0+I*0.0;
                channel2_out[mode_i*length + i] = 0.0+I*0.0;
                channel3_out[mode_i*length + i] = 0.0+I*0.0;
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

            //if (i %1000==0) printf("%e, %e, %e, %e, %e, %e\n", f, amp, phase, t0, tRef, phasetimeshift);


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

            channel1_out[mode_i*length + i] = ((transferL1_re+I*transferL1_im) * fastPart * X_ASD_inv[i]);

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

            channel2_out[mode_i*length + i] = ((transferL2_re+I*transferL2_im) * fastPart * Y_ASD_inv[i]);

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

            channel3_out[mode_i*length + i] = ((transferL3_re+I*transferL3_im) * fastPart * Z_ASD_inv[i]);
        }
    }
}*/

Interpolate::Interpolate(){
    int pass = 0;
}
void Interpolate::prep(double *b_mat, int m_, int n_, int to_gpu_){
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

    Interpolate::fit_constants(b_mat);
    delete[] d;
    delete[] dl;
    delete[] du;
    //dx_old = x_old[1] - x_old[0];
}

void Interpolate::fit_constants(double *b_mat){
    int i;
    double *w = new double[m];
    double *d_mat = new double[m];
    for (i=2; i<m; i++){
        //printf("%d\n", i);
        w[i] = dl[i]/d[i-1];
        d[i] = d[i] - w[i]*du[i-1];
        b_mat[i] = b_mat[i] - w[i]*b_mat[i-1];
        //printf("%lf, %lf, %lf\n", w[i], d[i], b[i]);
    }

    d_mat[m-1] = b_mat[m-1]/d[m-1];
    for (i=(m-2); i>=0; i--){
        d_mat[i] = (b_mat[i] - du[i]*d_mat[i+1])/d[i];
    }

    for (int i=0; i<m; i++) b_mat[i] = d_mat[i];
    delete[] d_mat;
    delete[] w;
}

Interpolate::~Interpolate(){
}
