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
void gpu_create_modes(ModeContainer *mode_vals, ModeContainer *h_mode_vals, int num_modes, int num_walkers, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp){
        //ModeContainer * cpu_mode_vals = cpu_create_modes(num_modes, num_walkers, l_vals, m_vals, max_length, 1, 0);

        int temp_modes = num_modes;
        num_modes = num_modes * num_walkers; // so we do not have to change the whole code.
        // This employs a special way to transfer structures with arrays to the GPU.
        int temp_i = 0;
        for (int mode_i=0; mode_i<num_modes; mode_i+=1){

            temp_i = mode_i % temp_modes;
            h_mode_vals[mode_i].num_modes = num_modes;
            h_mode_vals[mode_i].num_walkers = num_walkers;
            h_mode_vals[mode_i].l = l_vals[temp_i];
            h_mode_vals[mode_i].m = m_vals[temp_i];
            h_mode_vals[mode_i].max_length = max_length;
            h_mode_vals[mode_i].to_gpu = to_gpu;
            h_mode_vals[mode_i].to_interp = to_interp;

            // waveform
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].amp, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phase, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].freq_amp_phase, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].time_freq_corr, max_length*sizeof(double)));
            // response

            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].freq_response, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phaseRdelay, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_re, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_im, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_re, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_im, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_re, max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_im, max_length*sizeof(double)));

            if (to_interp == 1){
                // waveform
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].amp_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].amp_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].amp_coeff_3, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phase_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phase_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phase_coeff_3, (max_length-1)*sizeof(double)));

                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].time_freq_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].time_freq_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].time_freq_coeff_3, (max_length-1)*sizeof(double)));

                // response
                // transferL1
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_re_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_re_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_re_coeff_3, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_im_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_im_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL1_im_coeff_3, (max_length-1)*sizeof(double)));

                // transferL2
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_re_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_re_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_re_coeff_3, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_im_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_im_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL2_im_coeff_3, (max_length-1)*sizeof(double)));

                // transferL3
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_re_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_re_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_re_coeff_3, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_im_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_im_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].transferL3_im_coeff_3, (max_length-1)*sizeof(double)));

                // phasedelay
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phaseRdelay_coeff_1, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phaseRdelay_coeff_2, (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&h_mode_vals[mode_i].phaseRdelay_coeff_3, (max_length-1)*sizeof(double)));
            }

        }

        gpuErrchk(cudaMemcpy(mode_vals, h_mode_vals, num_modes*sizeof(ModeContainer), cudaMemcpyHostToDevice));

}

/*
Destroy ModeContainer structures.
*/
void gpu_destroy_modes(ModeContainer *d_mode_vals, ModeContainer *mode_vals){
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
}


WalkerContainer::WalkerContainer()
{}

void WalkerContainer::fill_info(int walker_i_, int num_modes_, unsigned int *l_vals_, unsigned int *m_vals_, int init_length_)
{

  walker_i = walker_i_;
  num_modes = num_modes_;

  init_length = init_length_;

  #ifdef __CUDACC__
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
    h_mode_vals = new ModeContainer[num_modes];
    gpuErrchk(cudaMalloc(&mode_vals, num_modes*sizeof(ModeContainer)));
    gpu_create_modes(mode_vals, h_mode_vals, num_modes, 1, l_vals_, m_vals_, init_length, 1, 1);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%e\n", milliseconds);
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
{}

void WalkerContainer::remove_info()
{

  #ifdef __CUDACC__
    gpu_destroy_modes(mode_vals, h_mode_vals);
    delete[] h_mode_vals;
    gpuErrchk(cudaFree(mode_vals));
    cudaFree(pHM_trans);
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
