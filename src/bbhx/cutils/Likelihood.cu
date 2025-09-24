#include "global.h"
#include "constants.h"
#include "Likelihood.hh"

#ifdef __CUDACC__
#include "cuComplex.h"
#include "cublas_v2.h"
#else
// #include <gsl/gsl_cblas.h>
#endif


#define  NUM_THREADS_LIKE 256

#define  DATA_BLOCK2 512

#ifdef __CUDACC__

#define NUM_THREADS_PREP 256

#ifdef __CUDACC__
CUDA_KERNEL
void new_hdyn_prep(cmplx *A0_out, cmplx *A1_out, cmplx *B0_out, cmplx *B1_out,
    cmplx *h0_arr, cmplx *data, double *psd, double *f_m_arr, double df, double *f_dense, int *data_index_all, int *noise_index_all,
    int *start_inds_all, int *num_points_seg, int length_f_rel, int num_bin, int data_length, int nchannels)
{

    __shared__ cmplx A0[NUM_THREADS_PREP];
    __shared__ cmplx A1[NUM_THREADS_PREP];
    __shared__ cmplx B0[NUM_THREADS_PREP];
    __shared__ cmplx B1[NUM_THREADS_PREP];

    int tid = threadIdx.x;

    int start_ind, num_points, seg_index, data_index, noise_index;
    int data_ind, noise_ind, template_ind, coefficient_index, ind;
    double Sn, f, f_m;
    cmplx d, h0, h0_conj;
    cmplx A0_tmp, A1_tmp, B0_tmp, B1_tmp, A0_flat, B0_flat;
    for (int bin_i = blockIdx.y; bin_i < num_bin; bin_i += gridDim.y)
    {
        data_index = data_index_all[bin_i];
        noise_index = noise_index_all[bin_i];
        for (int chan_i = blockIdx.z; chan_i < nchannels; chan_i += gridDim.z)
        {
            for (int seg_i = blockIdx.x; seg_i < length_f_rel - 1; seg_i += gridDim.x)
            {
                for (int i = threadIdx.x; i < blockDim.x; i += blockDim.x)
                {
                    A0[threadIdx.x] = 0.0;
                    A1[threadIdx.x] = 0.0;
                    B0[threadIdx.x] = 0.0;
                    B1[threadIdx.x] = 0.0;
                }
                __syncthreads();
                
                seg_index = bin_i * length_f_rel + seg_i + 1;
                coefficient_index = (bin_i * nchannels + chan_i) * length_f_rel + seg_i + 1;
                num_points = num_points_seg[seg_index];
                start_ind = start_inds_all[seg_index];
                f_m = f_m_arr[seg_index];

                A0_tmp = 0.0;
                A1_tmp = 0.0;
                B0_tmp = 0.0;
                B1_tmp = 0.0;

                // printf("%d %d %d %d %d\n", bin_i, chan_i, seg_i, length_f_rel, seg_index);
                for (int i = threadIdx.x; i < num_points; i += blockDim.x)
                {
                    ind = i + start_ind;
                    data_ind = (data_index * nchannels + chan_i) * data_length + ind;
                    noise_ind = (noise_index * nchannels + chan_i) * data_length + ind;
                    template_ind = (bin_i * nchannels + chan_i) * data_length + ind;

                    d = data[data_ind];
                    h0 = h0_arr[template_ind];
                    Sn = psd[noise_ind]; // inverse actually
                    f = f_dense[ind];

                    if ((ind > 400990) & (ind < 401000) & (data_index == 10))
                        printf("HUH: %d %d %d %d %e %e %e %e %e\n", ind, data_index, chan_i, nchannels, d.real(), d.imag(), h0.real(), h0.imag(), Sn);

                    h0_conj = gcmplx::conj(h0);

                    A0_flat = 4. * (h0_conj * d) * Sn * df;

                    B0_flat = 4. * (h0_conj * h0) * Sn * df;

                    A0_tmp += A0_flat;
                    A1_tmp += A0_flat * (f - f_m);
                    B0_tmp += B0_flat;
                    B1_tmp += B0_flat * (f - f_m);

                    // printf("check %e %e %e %e\n", A0_tmp.real(), A1_tmp.real(), B0_tmp.real(), B1_tmp.imag());                       
                }

                A0[threadIdx.x] = A0_tmp;
                A1[threadIdx.x] = A1_tmp;
                B0[threadIdx.x] = B0_tmp;
                B1[threadIdx.x] = B1_tmp;

                // if ((bin_i == 0) && (chan_i == 0) && (seg_i == 60)) printf("check %e %e %e %e\n", A0[threadIdx.x].real(), A1[threadIdx.x].real(), B0[threadIdx.x].real(), B1[threadIdx.x].real());

                __syncthreads();
                for (unsigned int s = 1; s < blockDim.x; s *= 2)
                {
                    if (tid % (2 * s) == 0)
                    {
                        A0[tid] += A0[tid + s];
                        A1[tid] += A1[tid + s];
                        B0[tid] += B0[tid + s];
                        B1[tid] += B1[tid + s];
                    }
                    __syncthreads();
                }
                __syncthreads();

                if (threadIdx.x == 0)
                {
                    A0_out[coefficient_index] = A0[0];
                    A1_out[coefficient_index] = A1[0];
                    B0_out[coefficient_index] = B0[0];
                    B1_out[coefficient_index] = B1[0];
                }
                __syncthreads();
            }
        }
    }
}

void new_hdyn_prep_wrap(cmplx *A0_out, cmplx *A1_out, cmplx *B0_out, cmplx *B1_out,
    cmplx *h0_arr, cmplx *data, double *psd, double *f_m_arr, double df, double *f_dense, int *data_index, int *noise_index,
    int *start_inds_all, int *num_points_seg, int length_f_rel, int num_bin, int data_length, int nchannels)
{
    dim3 grid(length_f_rel - 1, num_bin, nchannels);

    new_hdyn_prep<<<grid, NUM_THREADS_PREP>>>(A0_out, A1_out, B0_out, B1_out,
    h0_arr, data, psd, f_m_arr, df, f_dense, data_index, noise_index,
    start_inds_all, num_points_seg, length_f_rel, num_bin, data_length, nchannels);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

}

CUDA_KERNEL
void new_hdyn_like(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn, int *constants_index,
                    int numBinAll, int length_f_rel, int nChannels, int num_constants)
{
    
    int tid = threadIdx.x;
    __shared__ cmplx like1[NUM_THREADS_PREP];
    __shared__ cmplx like2[NUM_THREADS_PREP];
    cmplx A0, A1, B0, B1, rb, ra;
    cmplx r1, r0, r1Conj, tempLike1, tempLike2;
    double fb, fa, midFreq, mag_r0;
    int const_i;
    for (int bin_i = blockIdx.x; bin_i < numBinAll; bin_i += gridDim.x)
    {
        like1[tid] = 0.0;
        like2[tid] = 0.0;

        tempLike1 = 0.0;
        tempLike2 = 0.0;

        const_i = constants_index[bin_i];

        // sum all channels and i
        for (int chan_i = 0; chan_i < nChannels; chan_i += 1)
        {
            for (int i = threadIdx.x; i < length_f_rel - 1; i += blockDim.x)
            {
                A0 = dataConstants[((0 * num_constants + const_i) * nChannels + chan_i) * (length_f_rel) + i + 1];
                A1 = dataConstants[((1 * num_constants + const_i) * nChannels + chan_i) * (length_f_rel) + i + 1];
                B0 = dataConstants[((2 * num_constants + const_i) * nChannels + chan_i) * (length_f_rel) + i + 1];
                B1 = dataConstants[((3 * num_constants + const_i) * nChannels + chan_i) * (length_f_rel) + i + 1];

                fb = dataFreqsIn[const_i * length_f_rel + i + 1];
                fa = dataFreqsIn[const_i * length_f_rel + i];
                rb = templateChannels[(bin_i * nChannels + chan_i) * length_f_rel + i + 1];
                ra = templateChannels[(bin_i * nChannels + chan_i) * length_f_rel + i];
                // perform the actual computation

                // slope
                r1 = (rb - ra)/(fb - fa);
                midFreq = (fb + fa)/2.0;

                // intercept
                r0 = rb - r1 * (fb - midFreq);

                r1Conj = gcmplx::conj(r1);

                tempLike1 += A0 * gcmplx::conj(r0) + A1 * r1Conj;

                mag_r0 = gcmplx::abs(r0);
                tempLike2 += B0 * (mag_r0 * mag_r0) + 2. * B1 * gcmplx::real(r0 * r1Conj);
                
                // if (bin_i == 0) printf("%d %d %d %e %e %e %e %e %e %e %e\n", bin_i, chan_i, i, tempLike1.real(), tempLike2.real(), r1.real(), r0.real(), fb, fa, A0.real(), B0.real());
                
            }  
        }

        like1[tid] = tempLike1;
        like2[tid] = tempLike2;

        __syncthreads();
        for (unsigned int s = 1; s < blockDim.x; s *= 2)
        {
            if (tid % (2 * s) == 0)
            {
                like1[tid] += like1[tid + s];
                like2[tid] += like2[tid + s];
            }
            __syncthreads();
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            likeOut1[bin_i] = like1[0];
            likeOut2[bin_i] = like2[0];
        }
        __syncthreads();
    }
}


void new_hdyn_like_wrap(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn, int *constants_index,
                    int numBinAll, int length_f_rel, int nChannels, int num_constants)
{
    new_hdyn_like<<<numBinAll, NUM_THREADS_PREP>>>(likeOut1, likeOut2,
                    templateChannels, dataConstants,
                    dataFreqsIn, constants_index,
                    numBinAll, length_f_rel, nChannels, num_constants);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

#endif // __CUDACC__
// special way to run this. Need to separate CPU and GPU for this one
CUDA_KERNEL
void hdynLikelihood(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn,
                    int numBinAll, int data_length, int nChannels)
{

    // shared memory arrays for heterodyning coefficients
    __shared__ cmplx A0temp[DATA_BLOCK2];
    __shared__ cmplx A1temp[DATA_BLOCK2];
    __shared__ cmplx B0temp[DATA_BLOCK2];
    __shared__ cmplx B1temp[DATA_BLOCK2];
    __shared__ double dataFreqs[DATA_BLOCK2];

    // declare variables
    cmplx A0, A1, B0, B1;

    cmplx trans_complex(0.0, 0.0);
    cmplx prev_trans_complex(0.0, 0.0);
    double prevFreq = 0.0;
    double freq = 0.0;

    int currentStart = 0;

    cmplx r0, r1, r1Conj, tempLike1, tempLike2;
    double mag_r0, midFreq;

    int binNum = threadIdx.x + blockDim.x * blockIdx.x;

    tempLike1 = 0.0;
    tempLike2 = 0.0;
    // loop over channels
    for (int channel = 0; channel < nChannels; channel += 1)
    {
        // need to loop through frequencies and store the in shared memory carefully
        prevFreq = 0.0;
        currentStart = 0;
        while (currentStart < data_length)
        {
            __syncthreads();
            for (int jj = threadIdx.x; jj < DATA_BLOCK2; jj += blockDim.x)
            {
                // load in all the information for this group computation
                if ((jj + currentStart) >= data_length) continue;
                A0temp[jj] = dataConstants[(0 * nChannels + channel) * data_length + currentStart + jj];
                A1temp[jj] = dataConstants[(1 * nChannels + channel) * data_length + currentStart + jj];
                B0temp[jj] = dataConstants[(2 * nChannels + channel) * data_length + currentStart + jj];
                B1temp[jj] = dataConstants[(3 * nChannels + channel) * data_length + currentStart + jj];

                dataFreqs[jj] = dataFreqsIn[currentStart + jj];

            }
            __syncthreads();
            if (binNum < numBinAll)
            {
                for (int jj = 0; jj < DATA_BLOCK2; jj += 1)
                {
                    if ((jj + currentStart) >= data_length) continue;
                    freq = dataFreqs[jj];
                    trans_complex = templateChannels[((jj + currentStart) * nChannels + channel) * numBinAll + binNum];

                    // If we are after the first point
                    if ((prevFreq != 0.0) && (jj + currentStart > 0))
                    {
                        A0 = A0temp[jj]; // constants will need to be aligned with 1..n-1 because there are data_length - 1 bins
                        A1 = A1temp[jj];
                        B0 = B0temp[jj];
                        B1 = B1temp[jj];

                        // perform the actual computation

                        // slope
                        r1 = (trans_complex - prev_trans_complex)/(freq - prevFreq);
                        midFreq = (freq + prevFreq)/2.0;

                        // intercept
                        r0 = trans_complex - r1 * (freq - midFreq);

                        r1Conj = gcmplx::conj(r1);

                        tempLike1 += A0 * gcmplx::conj(r0) + A1 * r1Conj;

                        mag_r0 = gcmplx::abs(r0);
                        tempLike2 += B0 * (mag_r0 * mag_r0) + 2. * B1 * gcmplx::real(r0 * r1Conj);
                    }
                    // each step needs info from the last
                    prev_trans_complex = trans_complex;
                    prevFreq = freq;
                }
            }
            currentStart += DATA_BLOCK2;
        }
    }

    // Fill info
    if (binNum < numBinAll)
    {
        likeOut1[binNum] = tempLike1;
        likeOut2[binNum] = tempLike2;
    }
}

#else

// More straighforward on the CPU
void hdynLikelihood(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn,
                    int numBinAll, int data_length, int nChannels)
{

    #pragma omp parallel for
    for (int binNum = 0; binNum < numBinAll; binNum += 1)
    {
        cmplx A0, A1, B0, B1;

        cmplx trans_complex(0.0, 0.0);
        cmplx prev_trans_complex(0.0, 0.0);
        double prevFreq = 0.0;
        double freq = 0.0;

        cmplx r0, r1, r1Conj, tempLike1, tempLike2;
        double mag_r0, midFreq;

        tempLike1 = 0.0;
        tempLike2 = 0.0;

        for (int channel = 0; channel < nChannels; channel += 1)
        {
            prevFreq = 0.0;
            for (int jj = 0; jj < data_length - 1; jj += 1)
            {
                A0 = dataConstants[(0 * nChannels + channel) * data_length + jj];
                A1 = dataConstants[(1 * nChannels + channel) * data_length + jj];
                B0 = dataConstants[(2 * nChannels + channel) * data_length + jj];
                B1 = dataConstants[(3 * nChannels + channel) * data_length + jj];

                freq = dataFreqsIn[jj];

                trans_complex = templateChannels[((jj) * nChannels + channel) * numBinAll + binNum];

                if ((prevFreq != 0.0) && (jj > 0))
                {
                    r1 = (trans_complex - prev_trans_complex)/(freq - prevFreq);
                    midFreq = (freq + prevFreq)/2.0;

                    r0 = trans_complex - r1 * (freq - midFreq);

                    r1Conj = gcmplx::conj(r1);

                    tempLike1 += A0 * gcmplx::conj(r0) + A1 * r1Conj;

                    mag_r0 = gcmplx::abs(r0);
                    tempLike2 += B0 * (mag_r0 * mag_r0) + 2. * B1 * gcmplx::real(r0 * r1Conj);
                }

                prev_trans_complex = trans_complex;
                prevFreq = freq;
            }
        }
        likeOut1[binNum] = tempLike1;
        likeOut2[binNum] = tempLike2;
    }
}
#endif


void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels)
{

    int nblocks4 = std::ceil((numBinAll + NUM_THREADS_LIKE -1)/NUM_THREADS_LIKE);
    #ifdef __CUDACC__
    hdynLikelihood <<<nblocks4, NUM_THREADS_LIKE>>> (likeOut1, likeOut2, templateChannels, dataConstants, dataFreqs, numBinAll, data_length, nChannels);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
    hdynLikelihood(likeOut1, likeOut2, templateChannels, dataConstants, dataFreqs, numBinAll, data_length, nChannels);
    #endif
}

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


#define MAX_LENGTH_F_REL 512

// Not used any more but here in case fast computations of all coefficients are needed
// Needs to be checked
CUDA_KERNEL
void prep_hdyn(cmplx* A0_in, cmplx* A1_in, cmplx* B0_in, cmplx* B1_in, cmplx* d_arr, cmplx* h0_arr, double* S_n_arr, double df, int* bins, double* f_dense, double* f_m_arr, int data_length, int nchannels, int length_f_rel)
{

    CUDA_SHARED cmplx A0_temp[MAX_LENGTH_F_REL];
    CUDA_SHARED cmplx A1_temp[MAX_LENGTH_F_REL];
    CUDA_SHARED cmplx B0_temp[MAX_LENGTH_F_REL];
    CUDA_SHARED cmplx B1_temp[MAX_LENGTH_F_REL];

    int start, increment;
    for (int channel = 0; channel < nchannels; channel += 1)
    {
        CUDA_SYNC_THREADS;

        #ifdef __CUDACC__
        start = threadIdx.x;
        increment = blockDim.x;
        #else
        start = 0;
        increment = 1;
        #pragma omp parallel for
        #endif
        for (int i = start; i < length_f_rel - 1; i += increment)
        {
            A0_temp[i + 1] = 0.0;
            A1_temp[i + 1] = 0.0;
            B0_temp[i + 1] = 0.0;
            B1_temp[i + 1] = 0.0;
        }
        CUDA_SYNC_THREADS;

        #ifdef __CUDACC__
        start = threadIdx.x + blockDim.x * blockIdx.x;
        increment = blockDim.x * gridDim.x;
        #else
        start = 0;
        increment = 1;
        #pragma omp parallel for
        #endif
        for (int i = start; i < data_length; i += increment)
        {
            int bin_ind = bins[i];
            cmplx d = d_arr[channel * data_length + i];
            cmplx h0 = h0_arr[channel * data_length + i];
            double S_n = S_n_arr[channel * data_length + i];
            double f = f_dense[i];

            double f_m = f_m_arr[bin_ind];
            cmplx h0_conj = gcmplx::conj(h0);

            cmplx A0_flat = 4. * (h0_conj * d) / S_n * df;
            cmplx A1_flat = A0_flat * (f - f_m);

            cmplx B0_flat = 4. * (h0_conj * h0) / S_n * df;
            cmplx B1_flat = B0_flat * (f - f_m);
            #ifdef __CUDACC__
            atomicAddComplex(&A0_temp[bin_ind + 1], A0_flat);
            atomicAddComplex(&A1_temp[bin_ind + 1], A1_flat);
            atomicAddComplex(&B0_temp[bin_ind + 1], B0_flat);
            atomicAddComplex(&B1_temp[bin_ind + 1], B1_flat);
            #else
            #pragma omp critical
                A0_temp[bin_ind + 1] += A0_flat;
            #pragma omp critical
                A1_temp[bin_ind + 1] += A1_flat;
            #pragma omp critical
                B0_temp[bin_ind + 1] += B0_flat;
            #pragma omp critical
                B1_temp[bin_ind + 1] += B1_flat;
            #endif

        }

        CUDA_SYNC_THREADS;

        #ifdef __CUDACC__
        start = threadIdx.x;
        increment = blockDim.x;
        #else
        start = 0;
        increment = 1;
        #pragma omp parallel for
        #endif
        for (int i = start; i < length_f_rel - 1; i += increment)
        {
            #ifdef __CUDACC__
            atomicAddComplex(&A0_in[channel * length_f_rel + i + 1], A0_temp[i + 1]);
            atomicAddComplex(&A1_in[channel * length_f_rel + i + 1], A1_temp[i + 1]);
            atomicAddComplex(&B0_in[channel * length_f_rel + i + 1], B0_temp[i + 1]);
            atomicAddComplex(&B1_in[channel * length_f_rel + i + 1], B1_temp[i + 1]);
            #else
            #pragma omp critical
                A0_in[channel * length_f_rel + i + 1] += A0_temp[i + 1];
            #pragma omp critical
                A1_in[channel * length_f_rel + i + 1] += A1_temp[i + 1];
            #pragma omp critical
                B0_in[channel * length_f_rel + i + 1] += B0_temp[i + 1];
            #pragma omp critical
                B1_in[channel * length_f_rel + i + 1] += B1_temp[i + 1];
            #endif
        }
        CUDA_SYNC_THREADS;
    }
}

void prep_hdyn_wrap(cmplx* A0_in, cmplx* A1_in, cmplx* B0_in, cmplx* B1_in, cmplx* d_arr, cmplx* h0_arr, double* S_n_arr, double df, int* bins, double* f_dense, double* f_m_arr, int data_length, int nchannels, int length_f_rel)
{
    #ifdef __CUDACC__
    int nblocks = std::ceil((data_length + NUM_THREADS_LIKE -1)/NUM_THREADS_LIKE);
    prep_hdyn<<<nblocks, NUM_THREADS_LIKE>>>(A0_in, A1_in, B0_in, B1_in, d_arr, h0_arr, S_n_arr, df, bins, f_dense, f_m_arr, data_length, nchannels, length_f_rel);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
    prep_hdyn(A0_in, A1_in, B0_in, B1_in, d_arr, h0_arr, S_n_arr, df, bins, f_dense, f_m_arr, data_length, nchannels, length_f_rel);
    #endif
}

// add noise weighting efficiently to template
CUDA_KERNEL
void noiseweight_template(cmplx* templateChannels, double* noise_weight_times_df, int ind_start, int length, int data_stream_length, int nChannels)
{
    int start, increment;
    #ifdef __CUDACC__
    start = threadIdx.x + blockDim.x * blockIdx.x;
    increment = gridDim.x * blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int i = start; i < length; i += increment)
    {
        for (int j = 0; j < nChannels; j+= 1)
        {
            templateChannels[j * length + i] = templateChannels[j * length + i] * noise_weight_times_df[j * data_stream_length + ind_start + i];
        }
    }
}

#define NUM_THREADS_LIKE 256

// compute the likelihood directly
// different for CPU and GPU cause of streams
#ifdef __CUDACC__
void direct_like(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll, int nChannels, int device)
{
    // initialize everything
    cudaStream_t streams[numBinAll];
    cublasHandle_t handle;

    cuDoubleComplex result_d_h[numBinAll];
    cuDoubleComplex result_h_h[numBinAll];

    cudaSetDevice(device);
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf ("CUBLAS initialization failed\n");
      exit(0);
    }

    // omp over streams
    // TODO: can cause errors
    //#pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        // get information for this template
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        int nblocks = std::ceil((length_bin_i + NUM_THREADS_LIKE -1)/NUM_THREADS_LIKE);
        cudaStreamCreate(&streams[bin_i]);

        noiseweight_template
        <<<nblocks, NUM_THREADS_LIKE, 0, streams[bin_i]>>>(templateChannels, noise_weight_times_df, ind_start, length_bin_i, data_stream_length, nChannels);
        cudaStreamSynchronize(streams[bin_i]);

        for (int j = 0; j < nChannels; j += 1)
        {
            // setup cublas stream and run compuation in the desired frequency bounds
            double temp_real = 0.0;
            double temp_imag = 0.0;

            // d_h computation
            cublasSetStream(handle, streams[bin_i]);
            stat = cublasZdotc(handle, length_bin_i,
                              (cuDoubleComplex*)&dataChannels[j * data_stream_length + ind_start], 1,
                              (cuDoubleComplex*)&templateChannels[j * length_bin_i], 1,
                              &result_d_h[bin_i]);
            cudaStreamSynchronize(streams[bin_i]);
            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                exit(0);
            }

            temp_real = cuCreal(result_d_h[bin_i]);
            temp_imag = cuCimag(result_d_h[bin_i]);
            cmplx temp_d_h(temp_real, temp_imag);
            d_h[bin_i] += 4.0 * temp_d_h;

            // h_h computation
            cublasSetStream(handle, streams[bin_i]);
            stat = cublasZdotc(handle, length_bin_i,
                              (cuDoubleComplex*)&templateChannels[j * length_bin_i], 1,
                              (cuDoubleComplex*)&templateChannels[j * length_bin_i], 1,
                              &result_h_h[bin_i]);
            cudaStreamSynchronize(streams[bin_i]);
            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                exit(0);
            }

            temp_real = cuCreal(result_h_h[bin_i]);
            temp_imag = cuCimag(result_h_h[bin_i]);
            cmplx temp_h_h(temp_real, temp_imag);
            h_h[bin_i] += 4.0 * temp_h_h;

        }
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    //#pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
    cublasDestroy(handle);

}

#else

// matmul sub for cblas for backward compatibility
// FORTRAN STYLE COLUMN MAJOR (UGH)
template <typename T>
CUDA_CALLABLE_MEMBER void our_cblas_zdotc(
    int m,
    T *a,
    T *b,
    T *c)
{
    cmplx _out = 0.0;
    for (int i = 0; i < m; i++)
    {
       _out += gcmplx::conj(a[i]) * b[i];
    }
    *c = _out;
}

void direct_like(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll, int nChannels, int device)
{

    cmplx result_d_h[numBinAll];
    cmplx result_h_h[numBinAll];

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        noiseweight_template
        (templateChannels, noise_weight_times_df, ind_start, length_bin_i, data_stream_length, nChannels);

        for (int j = 0; j < nChannels; j += 1)
        {

            our_cblas_zdotc(length_bin_i,
                              &dataChannels[j * data_stream_length + ind_start],
                              &templateChannels[j * length_bin_i],
                              &result_d_h[bin_i]);

            d_h[bin_i] += 4.0 * result_d_h[bin_i];

            our_cblas_zdotc(length_bin_i,
                              &templateChannels[j * length_bin_i],
                              &templateChannels[j * length_bin_i],
                              &result_h_h[bin_i]);

            h_h[bin_i] += 4.0 * result_h_h[bin_i];
            //printf("%e %e\n", cuCreal(result_h_h[bin_i]), cuCreal(result_d_h[bin_i]));

        }
    }
}
#endif
