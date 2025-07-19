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
void TDILike(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* psd, double* dataFreqsIn, double* freqsOld, double* propArrays, double* c1In, double* c2In, double* c3In, int old_length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, int* inds, int ind_start, int ind_length, int bin_i, double df, int data_index, int num_data_sets, int noise_index, int num_noise_sets)
{

    __shared__ cmplx d_h_contrib[NUM_THREADS_BUILD];
    __shared__ cmplx h_h_contrib[NUM_THREADS_BUILD];

    int tid = threadIdx.x;

    for (int i = threadIdx.x; i < NUM_THREADS_BUILD; i += blockDim.x)
    {
        d_h_contrib[i] = 0.0;
        h_h_contrib[i] = 0.0;
    }
    __syncthreads();

    int start, increment;
    start = blockIdx.x * blockDim.x + threadIdx.x;
    increment = blockDim.x * gridDim.x;
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
        
        cmplx data_val1 = dataChannels[(data_index * 3 + 0) * data_length + (ind_start + i)];
        cmplx data_val2 = dataChannels[(data_index * 3 + 1) * data_length + (ind_start + i)];
        cmplx data_val3 = dataChannels[(data_index * 3 + 2) * data_length + (ind_start + i)];

        double psd_val1 = psd[(noise_index * 3 + 0) * data_length + (ind_start + i)];
        double psd_val2 = psd[(noise_index * 3 + 1) * data_length + (ind_start + i)];
        double psd_val3 = psd[(noise_index * 3 + 2) * data_length + (ind_start + i)];

        // cmplx _tmp1 = 4.0 * df * (gcmplx::conj(trans_complex1) * trans_complex1 / psd_val1);
        // if ((bin_i == 0) && (f > 0.03) && (f < 0.0301))
        //     printf("Check %d %d %d %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", i, numModes, ind_start, (0 * num_data_sets + noise_index) * data_length + (ind_start + i), f, trans_complex1.real(), trans_complex1.imag(), data_val1.real(), data_val1.imag(), psd_val1, _tmp1.real(), _tmp1.imag());

        d_h_contrib[tid] += 4.0 * df * (gcmplx::conj(data_val1) * trans_complex1 / psd_val1);
        d_h_contrib[tid] += 4.0 * df * (gcmplx::conj(data_val2) * trans_complex2 / psd_val2);
        d_h_contrib[tid] += 4.0 * df * (gcmplx::conj(data_val3) * trans_complex3 / psd_val3);

        // if ((blockIdx.x >= 3) && (blockIdx.x <= 5))
        //     printf("%d %d %e %e %e %e %e\n", blockIdx.x, tid, (4.0 * df * (gcmplx::conj(trans_complex1) * trans_complex1 / psd_val1)).real(), df, trans_complex1.real(), trans_complex1.imag(), psd_val1);
        
        h_h_contrib[tid] += 4.0 * df * (gcmplx::conj(trans_complex1) * trans_complex1 / psd_val1);
        h_h_contrib[tid] += 4.0 * df * (gcmplx::conj(trans_complex2) * trans_complex2 / psd_val2);
        h_h_contrib[tid] += 4.0 * df * (gcmplx::conj(trans_complex3) * trans_complex3 / psd_val3);

        // dataChannels[(ind_start + i)] = 4.0 * df * (gcmplx::conj(trans_complex1) * trans_complex1 / psd_val1);
    }
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            d_h_contrib[tid] += d_h_contrib[tid + s];
            h_h_contrib[tid] += h_h_contrib[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();

    if (tid == 0)
    {
        atomicAddComplex(&d_h[bin_i], d_h_contrib[0]);
        atomicAddComplex(&h_h[bin_i], h_h_contrib[0]);
        // printf("hh contrib: %d %e\n", blockIdx.x, h_h_contrib[0].real());
    }
    __syncthreads();
}


void InterpTDILike(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* psd, double* dataFreqs, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_start_in, double* t_end_in, int length, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths, double df, int* data_index_all, int num_data_sets, int* noise_index_all, int num_noise_sets, int gpu)
{

    gpuErrchk(cudaSetDevice(gpu));

    cudaStream_t streams[numBinAll];

    // interpolation is done in streams on GPU
    // #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {

        // get all information ready included casting pointers properly
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];
        int* inds = (int*) inds_ptrs[bin_i];

        double t_start = t_start_in[bin_i];
        double t_end = t_end_in[bin_i];

        int data_index = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        int nblocks3 = std::ceil((length_bin_i + NUM_THREADS_BUILD -1)/NUM_THREADS_BUILD);

        //#ifdef __CUDACC__
        dim3 gridDim(nblocks3, 1);
        gpuErrchk(cudaStreamCreate(&streams[bin_i]));
        TDILike<<<gridDim, NUM_THREADS_BUILD, 0, streams[bin_i]>>>(d_h, h_h, dataChannels, psd, dataFreqs, freqs, propArrays, c1, c2, c3, length, data_length, numBinAll, numModes, t_start, t_end, inds, ind_start, length_bin_i, bin_i, df, data_index, num_data_sets, noise_index, num_noise_sets);
        //#else
        //TDILike(templateChannels, dataFreqs, freqs, propArrays, c1, c2, c3, length, data_length, numBinAll, numModes, t_start, t_end, inds, ind_start, /length_bin_i, bin_i);
        //#endif

    }

   //#ifdef __CUDACC__
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        gpuErrchk(cudaStreamDestroy(streams[bin_i]));
    }
    //#endif
}

