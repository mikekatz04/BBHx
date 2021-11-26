#include "constants.h"
#include "global.h"
#include "WaveformBuild.hh"


#define NUM_THREADS_BUILD 256

#ifdef __CUDACC__
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void atomicAddComplex(cmplx* a, cmplx b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAddDouble(x, b.real());
  atomicAddDouble(y, b.imag());
}
#endif



#define  DATA_BLOCK 128
#define  NUM_INTERPS 9

CUDA_CALLABLE_MEMBER
cmplx get_ampphasefactor(double amp, double phase, double phaseShift){
    return amp*gcmplx::exp(cmplx(0.0, phase + phaseShift));
}

CUDA_CALLABLE_MEMBER
void combine_information(cmplx* channel1, cmplx* channel2, cmplx* channel3, double amp, double phase, double tf, cmplx transferL1, cmplx transferL2, cmplx transferL3, double t_start, double t_end)
{
    if (((tf >= t_start)) && ((tf <= t_end) || (t_end <= 0.0)))
    {
        // this is the final waveform combination
        // only happens if it is in the time bounds
        cmplx amp_phase_term = amp*gcmplx::exp(cmplx(0.0, phase));

        *channel1 = gcmplx::conj(transferL1 * amp_phase_term);
        *channel2 = gcmplx::conj(transferL2 * amp_phase_term);
        *channel3 = gcmplx::conj(transferL3 * amp_phase_term);

    }
}

#define  NUM_TERMS 4

#define  MAX_NUM_COEFF_TERMS 1200

// interpolate to TDI channels
CUDA_KERNEL
void TDI(cmplx* templateChannels, double* dataFreqsIn, double* freqsOld, double* propArrays, double* c1In, double* c2In, double* c3In, int old_length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, int* inds, int ind_start, int ind_length, int bin_i)
{

    int start, increment;
    #ifdef __CUDACC__
    start = blockIdx.x * blockDim.x + threadIdx.x;
    increment = blockDim.x *gridDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int i = start; i < ind_length; i += increment)
    {
        // get x information for this spline evaluation
        double f = dataFreqsIn[i + ind_start];

        int ind_here = inds[i];

        double f_old = freqsOld[bin_i * old_length + ind_here];

        double x = f - f_old;
        double x2 = x * x;
        double x3 = x * x2;

        cmplx trans_complex1 = 0.0; cmplx trans_complex2 = 0.0; cmplx trans_complex3 = 0.0;

        for (int mode_i = 0; mode_i < numModes; mode_i += 1)
        {
            // evaluate all spline quantities

            int int_shared = ((0 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double amp = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((1 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double phase = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((2 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double tf = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((3 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double transferL1_re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((4 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double transferL1_im = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((5 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double transferL2_re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((6 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double transferL2_im = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((7 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double transferL3_re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            int_shared = ((8 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
            double transferL3_im = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

            cmplx channel1(0.0, 0.0);
            cmplx channel2(0.0, 0.0);
            cmplx channel3(0.0, 0.0);

            combine_information(&channel1, &channel2, &channel3, amp, phase, tf, cmplx(transferL1_re, transferL1_im), cmplx(transferL2_re, transferL2_im), cmplx(transferL3_re, transferL3_im), t_obs_start, t_obs_end);

            // add all modes together directly
            trans_complex1 += channel1;
            trans_complex2 += channel2;
            trans_complex3 += channel3;
        }

        templateChannels[0 * ind_length + i] = trans_complex1;
        templateChannels[1 * ind_length + i] = trans_complex2;
        templateChannels[2 * ind_length + i] = trans_complex3;
    }
}


void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_start_in, double* t_end_in, int length, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths)
{
    #ifdef __CUDACC__
    cudaStream_t streams[numBinAll];
    #endif

    // interpolation is done in streams on GPU
    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        // get all information ready included casting pointers properly
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];
        int* inds = (int*) inds_ptrs[bin_i];

        double t_start = t_start_in[bin_i];
        double t_end = t_end_in[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        int nblocks3 = std::ceil((length_bin_i + NUM_THREADS_BUILD -1)/NUM_THREADS_BUILD);

        #ifdef __CUDACC__
        dim3 gridDim(nblocks3, 1);
        cudaStreamCreate(&streams[bin_i]);
        TDI<<<gridDim, NUM_THREADS_BUILD, 0, streams[bin_i]>>>(templateChannels, dataFreqs, freqs, propArrays, c1, c2, c3, length, data_length, numBinAll, numModes, t_start, t_end, inds, ind_start, length_bin_i, bin_i);
        #else
        TDI(templateChannels, dataFreqs, freqs, propArrays, c1, c2, c3, length, data_length, numBinAll, numModes, t_start, t_end, inds, ind_start, length_bin_i, bin_i);
        #endif

    }

    #ifdef __CUDACC__
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
    #endif
}


// directly fill waveform with no interpolation
// parallel method here is one block per binary
CUDA_KERNEL
void fill_waveform(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)
{

    cmplx I(0.0, 1.0);

    cmplx temp_channel1 = 0.0, temp_channel2 = 0.0, temp_channel3 = 0.0;
    int start, increment;
    #ifdef __CUDACC__
    start = blockIdx.x;
    increment = gridDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int bin_i = start; bin_i < numBinAll; bin_i += increment)
    {

        double t_start_bin = t_start[bin_i];
        double t_end_bin = t_end[bin_i];

        int start2, increment2;
        #ifdef __CUDACC__
        start2 = threadIdx.x;
        increment2 = blockDim.x;
        #else
        start2 = 0;
        increment2 = 1;
        #pragma omp parallel for
        #endif
        for (int i = start2; i < data_length; i += increment2)
        {
            cmplx temp_channel1 = 0.0;
            cmplx temp_channel2 = 0.0;
            cmplx temp_channel3 = 0.0;
            for (int mode_i = 0; mode_i < numModes; mode_i += 1)
            {

                // get each value directly out of the holder arrays

                int ind = ((0 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double amp = bbh_buffer[ind];

                ind = ((1 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double phase = bbh_buffer[ind];

                ind = ((2 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double tf = bbh_buffer[ind];

                ind = ((3 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL1_re = bbh_buffer[ind];

                ind = ((4 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL1_im = bbh_buffer[ind];

                ind = ((5 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL2_re = bbh_buffer[ind];

                ind = ((6 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL2_im = bbh_buffer[ind];

                ind = ((7 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL3_re = bbh_buffer[ind];

                ind = ((8 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL3_im = bbh_buffer[ind];

                cmplx channel1 = 0.0 + 0.0 * I;
                cmplx channel2 = 0.0 + 0.0 * I;
                cmplx channel3 = 0.0 + 0.0 * I;

                combine_information(&channel1, &channel2, &channel3, amp, phase, tf, cmplx(transferL1_re, transferL1_im), cmplx(transferL2_re, transferL2_im), cmplx(transferL3_re, transferL3_im), t_start_bin, t_end_bin);

                temp_channel1 += channel1;
                temp_channel2 += channel2;
                temp_channel3 += channel3;
            }

            templateChannels[(bin_i * 3 + 0) * data_length + i] = temp_channel1;
            templateChannels[(bin_i * 3 + 1) * data_length + i] = temp_channel2;
            templateChannels[(bin_i * 3 + 2) * data_length + i] = temp_channel3;

        }
    }
}

void direct_sum(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)
{

    // block per binary
    int nblocks5 = numBinAll;

    #ifdef __CUDACC__
    fill_waveform<<<nblocks5, NUM_THREADS_BUILD>>>(templateChannels, bbh_buffer, numBinAll, data_length, nChannels, numModes, t_start, t_end);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
    fill_waveform(templateChannels, bbh_buffer, numBinAll, data_length, nChannels, numModes, t_start, t_end);
    #endif
}
