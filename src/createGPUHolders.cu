/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  Creates the structures that hold waveform and interpolation information
 *  for the GPU version of the PhenomHM waveform.
 *
 *
 *  Copyright (C) 2019 Michael Katz
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"
#include "stdio.h"
#include "createGPUHolders.hh"

/*
Function for creating ModeContainer on the gpu.
*/
ModeContainer * gpu_create_modes(int num_modes, int num_walkers, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp){
        ModeContainer * cpu_mode_vals = cpu_create_modes(num_modes, num_walkers, l_vals, m_vals, max_length, 1, 0);
        ModeContainer * mode_vals;

        num_modes = num_modes * num_walkers; // so we do not have to change the whole code.
        // This employs a special way to transfer structures with arrays to the GPU.

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

/*
Destroy ModeContainer structures.
*/
void gpu_destroy_modes(ModeContainer * mode_vals){
    for (int i=0; i<(mode_vals[0].num_modes*mode_vals[0].num_walkers); i++){
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


WalkerContainer::WalkerContainer(int walker_i_, int num_modes_, unsigned int *l_vals_, unsigned int *m_vals_, int init_length_)
{

  walker_i = walker_i_;
  num_modes_ = num_modes;

  init_length = init_length_;

  #ifdef __CUDACC__
    mode_vals = gpu_create_modes(num_modes, 1, l_vals_, m_vals_, init_length, 1, 1);
    cudaMalloc(&pHM_trans, sizeof(PhenomHMStorage));
    cudaMalloc(&pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients));
    cudaMalloc(&amp_prefactors_trans, sizeof(AmpInsPrefactors));
    cudaMalloc(&pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp));
    cudaMalloc(&q_all_trans, num_modes*sizeof(HMPhasePreComp));


  #else

    mode_vals = create_cpu_modes(num_modes, 1, l_vals_, m_vals_, init_length, 0, 1)
    pHM_trans = new PhenomHMStorage;
    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;
    amp_prefactors_trans = new AmpInsPrefactors;
    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];
    q_all_trans = new HMPhasePreComp[num_modes];

  #endif

}

WalkerContainer::~WalkerContainer()
{

  #ifdef __CUDACC__
    gpu_destroy_modes(mode_vals);
    cudafree(pHM_trans);
    cudaFree(pAmp_trans);
    cudaFree(amp_prefactors_trans);
    cudaFree(pDPreComp_all_trans);
    cudaFree(q_all_trans);


  #else

    cpu_destroy_modes(mode_vals);
    delete pHM_trans;
    delete pAmp_trans;
    delete amp_prefactors_trans;
    delete[] pDPreComp_all_trans;
    delete[] q_all_trans;

  #endif

}
