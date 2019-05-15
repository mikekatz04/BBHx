#include "globalPhenomHM.h"


ModeContainer * cpu_create_modes(int num_modes, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp){
        ModeContainer * mode_vals = new ModeContainer[num_modes];
        for (int i=0; i<num_modes; i++){
            mode_vals[i].num_modes = num_modes;
            mode_vals[i].l = l_vals[i];
            mode_vals[i].m = m_vals[i];
            mode_vals[i].max_length = max_length;
            mode_vals[i].to_gpu = to_gpu;
            mode_vals[i].to_interp = to_interp;

                mode_vals[i].amp = new double[max_length];
                mode_vals[i].phase = new double[max_length];
                mode_vals[i].freq_amp_phase = new double[max_length];
                mode_vals[i].time_freq_corr = new double[max_length];

                mode_vals[i].freq_response = new double[max_length];
                mode_vals[i].phaseRdelay = new double[max_length];

                mode_vals[i].transferL1_re = new double[max_length];
                mode_vals[i].transferL1_im = new double[max_length];

                mode_vals[i].transferL2_re = new double[max_length];
                mode_vals[i].transferL2_im = new double[max_length];

                mode_vals[i].transferL3_re = new double[max_length];
                mode_vals[i].transferL3_im = new double[max_length];

            if (mode_vals[i].to_gpu == 0){
                if (mode_vals[i].to_interp == 1){
                    mode_vals[i].amp_coeff_1 = new double[max_length -1];
                    mode_vals[i].amp_coeff_2 = new double[max_length -1];
                    mode_vals[i].amp_coeff_3 = new double[max_length -1];
                    mode_vals[i].phase_coeff_1 = new double[max_length -1];
                    mode_vals[i].phase_coeff_2 = new double[max_length -1];
                    mode_vals[i].phase_coeff_3 = new double[max_length -1];

                    mode_vals[i].time_freq_coeff_1 = new double[max_length -1];
                    mode_vals[i].time_freq_coeff_2 = new double[max_length -1];
                    mode_vals[i].time_freq_coeff_3 = new double[max_length -1];

                    mode_vals[i].transferL1_re_coeff_1 = new double[max_length -1];
                    mode_vals[i].transferL1_re_coeff_2 = new double[max_length -1];
                    mode_vals[i].transferL1_re_coeff_3 = new double[max_length -1];
                    mode_vals[i].transferL1_im_coeff_1 = new double[max_length -1];
                    mode_vals[i].transferL1_im_coeff_2 = new double[max_length -1];
                    mode_vals[i].transferL1_im_coeff_3 = new double[max_length -1];

                    mode_vals[i].transferL2_re_coeff_1 = new double[max_length -1];
                    mode_vals[i].transferL2_re_coeff_2 = new double[max_length -1];
                    mode_vals[i].transferL2_re_coeff_3 = new double[max_length -1];
                    mode_vals[i].transferL2_im_coeff_1 = new double[max_length -1];
                    mode_vals[i].transferL2_im_coeff_2 = new double[max_length -1];
                    mode_vals[i].transferL2_im_coeff_3 = new double[max_length -1];

                    mode_vals[i].transferL3_re_coeff_1 = new double[max_length -1];
                    mode_vals[i].transferL3_re_coeff_2 = new double[max_length -1];
                    mode_vals[i].transferL3_re_coeff_3 = new double[max_length -1];
                    mode_vals[i].transferL3_im_coeff_1 = new double[max_length -1];
                    mode_vals[i].transferL3_im_coeff_2 = new double[max_length -1];
                    mode_vals[i].transferL3_im_coeff_3 = new double[max_length -1];

                    mode_vals[i].phaseRdelay_coeff_1 = new double[max_length -1];
                    mode_vals[i].phaseRdelay_coeff_2 = new double[max_length -1];
                    mode_vals[i].phaseRdelay_coeff_3 = new double[max_length -1];

                }
            }
        }
        return mode_vals;
}

void cpu_destroy_modes(ModeContainer * mode_vals){
    if (mode_vals[0].to_gpu == 0){
        for (int i=0; i<mode_vals[0].num_modes; i++){
            delete[] mode_vals[i].amp;
            delete[] mode_vals[i].phase;
            delete[] mode_vals[i].freq_amp_phase;
            delete[] mode_vals[i].time_freq_corr;

            delete[] mode_vals[i].freq_response;
            delete[] mode_vals[i].phaseRdelay;

            delete[] mode_vals[i].transferL1_re;
            delete[] mode_vals[i].transferL1_im;

            delete[] mode_vals[i].transferL2_re;
            delete[] mode_vals[i].transferL2_im;

            delete[] mode_vals[i].transferL3_re;
            delete[] mode_vals[i].transferL3_im;

            if (mode_vals[i].to_interp == 1){
                delete[] mode_vals[i].amp_coeff_1;
                delete[] mode_vals[i].amp_coeff_2;
                delete[] mode_vals[i].amp_coeff_3;
                delete[] mode_vals[i].phase_coeff_1;
                delete[] mode_vals[i].phase_coeff_2;
                delete[] mode_vals[i].phase_coeff_3;
                delete[] mode_vals[i].time_freq_coeff_1;
                delete[] mode_vals[i].time_freq_coeff_2;
                delete[] mode_vals[i].time_freq_coeff_3;

                delete[] mode_vals[i].transferL1_re_coeff_1;
                delete[] mode_vals[i].transferL1_re_coeff_2;
                delete[] mode_vals[i].transferL1_re_coeff_3;
                delete[] mode_vals[i].transferL1_im_coeff_1;
                delete[] mode_vals[i].transferL1_im_coeff_2;
                delete[] mode_vals[i].transferL1_im_coeff_3;

                delete[] mode_vals[i].transferL2_re_coeff_1;
                delete[] mode_vals[i].transferL2_re_coeff_2;
                delete[] mode_vals[i].transferL2_re_coeff_3;
                delete[] mode_vals[i].transferL2_im_coeff_1;
                delete[] mode_vals[i].transferL2_im_coeff_2;
                delete[] mode_vals[i].transferL2_im_coeff_3;

                delete[] mode_vals[i].transferL3_re_coeff_1;
                delete[] mode_vals[i].transferL3_re_coeff_2;
                delete[] mode_vals[i].transferL3_re_coeff_3;
                delete[] mode_vals[i].transferL3_im_coeff_1;
                delete[] mode_vals[i].transferL3_im_coeff_2;
                delete[] mode_vals[i].transferL3_im_coeff_3;

                delete[] mode_vals[i].phaseRdelay_coeff_1;
                delete[] mode_vals[i].phaseRdelay_coeff_2;
                delete[] mode_vals[i].phaseRdelay_coeff_3;
            }
        }
    }
    delete[] mode_vals;
}
