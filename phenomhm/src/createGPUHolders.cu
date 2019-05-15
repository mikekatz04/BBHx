#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


ModeContainer * gpu_create_modes(int num_modes, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp){
        ModeContainer * cpu_mode_vals = cpu_create_modes(num_modes,  l_vals, m_vals, max_length, 1, 0);
        ModeContainer * mode_vals;

        double *amp[num_modes];
        double *phase[num_modes];
        double *freq_amp_phase[num_modes];
        double *time_freq_corr[num_modes];

        double *amp_coeff_1[num_modes];
        double *amp_coeff_2[num_modes];
        double *amp_coeff_3[num_modes];

        double *phase_coeff_1[num_modes];
        double *phase_coeff_2[num_modes];
        double *phase_coeff_3[num_modes];

        double *time_freq_coeff_1[num_modes];
        double *time_freq_coeff_2[num_modes];
        double *time_freq_coeff_3[num_modes];

        double *freq_response[num_modes];

        double *phaseRdelay[num_modes];
        double *phaseRdelay_coeff_1[num_modes];
        double *phaseRdelay_coeff_2[num_modes];
        double *phaseRdelay_coeff_3[num_modes];

        double *transferL1_re[num_modes];
        double *transferL1_im[num_modes];
        double *transferL1_re_coeff_1[num_modes];
        double *transferL1_re_coeff_2[num_modes];
        double *transferL1_re_coeff_3[num_modes];
        double *transferL1_im_coeff_1[num_modes];
        double *transferL1_im_coeff_2[num_modes];
        double *transferL1_im_coeff_3[num_modes];

        double *transferL2_re[num_modes];
        double *transferL2_im[num_modes];
        double *transferL2_re_coeff_1[num_modes];
        double *transferL2_re_coeff_2[num_modes];
        double *transferL2_re_coeff_3[num_modes];
        double *transferL2_im_coeff_1[num_modes];
        double *transferL2_im_coeff_2[num_modes];
        double *transferL2_im_coeff_3[num_modes];

        double *transferL3_re[num_modes];
        double *transferL3_im[num_modes];
        double *transferL3_re_coeff_1[num_modes];
        double *transferL3_re_coeff_2[num_modes];
        double *transferL3_re_coeff_3[num_modes];
        double *transferL3_im_coeff_1[num_modes];
        double *transferL3_im_coeff_2[num_modes];
        double *transferL3_im_coeff_3[num_modes];


        gpuErrchk(cudaMalloc(&mode_vals, num_modes*sizeof(ModeContainer)));
        gpuErrchk(cudaMemcpy(mode_vals, cpu_mode_vals, num_modes*sizeof(ModeContainer), cudaMemcpyHostToDevice));

        for (int i=0; i<num_modes; i++){

            // waveform
            gpuErrchk(cudaMalloc(&amp[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&phase[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&freq_amp_phase[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&time_freq_corr[i], max_length*sizeof(double)));

            gpuErrchk(cudaMemcpy(&(mode_vals[i].amp), &(amp[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].phase), &(phase[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].freq_amp_phase), &(freq_amp_phase[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].time_freq_corr), &(time_freq_corr[i]), sizeof(double *), cudaMemcpyHostToDevice));

            // response

            gpuErrchk(cudaMalloc(&freq_response[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&phaseRdelay[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&transferL1_re[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&transferL1_im[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&transferL2_re[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&transferL2_im[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&transferL3_re[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&transferL3_im[i], max_length*sizeof(double)));


            gpuErrchk(cudaMemcpy(&(mode_vals[i].freq_response), &(freq_response[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].phaseRdelay), &(phaseRdelay[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_re), &(transferL1_re[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_im), &(transferL1_im[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_re), &(transferL2_re[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_im), &(transferL2_im[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_re), &(transferL3_re[i]), sizeof(double *), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_im), &(transferL3_im[i]), sizeof(double *), cudaMemcpyHostToDevice));

            if (to_interp == 1){
                // waveform
                gpuErrchk(cudaMalloc(&amp_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&amp_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&amp_coeff_3[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phase_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phase_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phase_coeff_3[i], (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMalloc(&time_freq_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&time_freq_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&time_freq_coeff_3[i], (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].amp_coeff_1), &(amp_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].amp_coeff_2), &(amp_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].amp_coeff_3), &(amp_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].phase_coeff_1), &(phase_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].phase_coeff_2), &(phase_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].phase_coeff_3), &(phase_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].time_freq_coeff_1), &(time_freq_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].time_freq_coeff_2), &(time_freq_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].time_freq_coeff_3), &(time_freq_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));

                // response
                // transferL1
                gpuErrchk(cudaMalloc(&transferL1_re_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL1_re_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL1_re_coeff_3[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL1_im_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL1_im_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL1_im_coeff_3[i], (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_re_coeff_1), &(transferL1_re_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_re_coeff_2), &(transferL1_re_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_re_coeff_3), &(transferL1_re_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_im_coeff_1), &(transferL1_im_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_im_coeff_2), &(transferL1_im_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL1_im_coeff_3), &(transferL1_im_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));

                // transferL2
                gpuErrchk(cudaMalloc(&transferL2_re_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL2_re_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL2_re_coeff_3[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL2_im_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL2_im_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL2_im_coeff_3[i], (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_re_coeff_1), &(transferL2_re_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_re_coeff_2), &(transferL2_re_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_re_coeff_3), &(transferL2_re_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_im_coeff_1), &(transferL2_im_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_im_coeff_2), &(transferL2_im_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL2_im_coeff_3), &(transferL2_im_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));

                // transferL3
                gpuErrchk(cudaMalloc(&transferL3_re_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL3_re_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL3_re_coeff_3[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL3_im_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL3_im_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&transferL3_im_coeff_3[i], (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_re_coeff_1), &(transferL3_re_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_re_coeff_2), &(transferL3_re_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_re_coeff_3), &(transferL3_re_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_im_coeff_1), &(transferL3_im_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_im_coeff_2), &(transferL3_im_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].transferL3_im_coeff_3), &(transferL3_im_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));

                // phasedelay
                gpuErrchk(cudaMalloc(&phaseRdelay_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phaseRdelay_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phaseRdelay_coeff_3[i], (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMemcpy(&(mode_vals[i].phaseRdelay_coeff_1), &(phaseRdelay_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].phaseRdelay_coeff_2), &(phaseRdelay_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(&(mode_vals[i].phaseRdelay_coeff_3), &(phaseRdelay_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice));
            }
        }

        return mode_vals;
}

void gpu_destroy_modes(ModeContainer * mode_vals){
    for (int i=0; i<mode_vals[0].num_modes; i++){
        gpuErrchk(cudaFree(mode_vals[i].amp));
        gpuErrchk(cudaFree(mode_vals[i].phase));
        gpuErrchk(cudaFree(mode_vals[i].freq_amp_phase));
        gpuErrchk(cudaFree(mode_vals[i].time_freq_corr));

        gpuErrchk(cudaFree(mode_vals[i].freq_response));
        gpuErrchk(cudaFree(mode_vals[i].transferL1_re));
        gpuErrchk(cudaFree(mode_vals[i].transferL1_im));
        gpuErrchk(cudaFree(mode_vals[i].transferL2_re));
        gpuErrchk(cudaFree(mode_vals[i].transferL2_im));
        gpuErrchk(cudaFree(mode_vals[i].transferL3_re));
        gpuErrchk(cudaFree(mode_vals[i].transferL3_im));
        gpuErrchk(cudaFree(mode_vals[i].phaseRdelay));

        //gpuErrchk(cudaFree(mode_vals[i].hI));
        //gpuErrchk(cudaFree(mode_vals[i].hII));
        if (mode_vals[i].to_interp == 1){
            gpuErrchk(cudaFree(mode_vals[i].amp_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].amp_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].amp_coeff_3));

            gpuErrchk(cudaFree(mode_vals[i].phase_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].phase_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].phase_coeff_3));

            gpuErrchk(cudaFree(mode_vals[i].time_freq_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].time_freq_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].time_freq_coeff_3));

            // transferL1
            gpuErrchk(cudaFree(mode_vals[i].transferL1_re_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].transferL1_re_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].transferL1_re_coeff_3));
            gpuErrchk(cudaFree(mode_vals[i].transferL1_im_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].transferL1_im_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].transferL1_im_coeff_3));

            // transferL2
            gpuErrchk(cudaFree(mode_vals[i].transferL2_re_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].transferL2_re_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].transferL2_re_coeff_3));
            gpuErrchk(cudaFree(mode_vals[i].transferL2_im_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].transferL2_im_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].transferL2_im_coeff_3));

            // transferL3
            gpuErrchk(cudaFree(mode_vals[i].transferL3_re_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].transferL3_re_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].transferL3_re_coeff_3));
            gpuErrchk(cudaFree(mode_vals[i].transferL3_im_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].transferL3_im_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].transferL3_im_coeff_3));

            // phasedelay
            gpuErrchk(cudaFree(mode_vals[i].phaseRdelay_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].phaseRdelay_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].phaseRdelay_coeff_3));
        }
    }
    gpuErrchk(cudaFree(mode_vals));
}
