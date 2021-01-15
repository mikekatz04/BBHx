#include "global.h"
#include "constants.h"
#include "Likelihood.hh"

#ifdef __CUDACC__
#include "cuComplex.h"
#include "cublas_v2.h"
#else
#include <gsl/gsl_cblas.h>
#endif

#define  NUM_THREADS_LIKE 256

#define  DATA_BLOCK2 512

#ifdef __CUDACC__
CUDA_KERNEL
void hdynLikelihood(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn,
                    int numBinAll, int data_length, int nChannels)
{
    __shared__ cmplx A0temp[DATA_BLOCK2];
    __shared__ cmplx A1temp[DATA_BLOCK2];
    __shared__ cmplx B0temp[DATA_BLOCK2];
    __shared__ cmplx B1temp[DATA_BLOCK2];
    __shared__ double dataFreqs[DATA_BLOCK2];

    cmplx A0, A1, B0, B1;

    cmplx trans_complex(0.0, 0.0);
    cmplx prev_trans_complex(0.0, 0.0);
    double prevFreq = 0.0;
    double freq = 0.0;

    int currentStart = 0;

    cmplx r0, r1, r1Conj, tempLike1, tempLike2;
    double mag_r0, midFreq;

    int binNum = threadIdx.x + blockDim.x * blockIdx.x;

    if (true) // for (int binNum = threadIdx.x + blockDim.x * blockIdx.x; binNum < numBinAll; binNum += blockDim.x * gridDim.x)
    {
        tempLike1 = 0.0;
        tempLike2 = 0.0;
        for (int channel = 0; channel < nChannels; channel += 1)
        {
            prevFreq = 0.0;
            currentStart = 0;
            while (currentStart < data_length)
            {
                __syncthreads();
                for (int jj = threadIdx.x; jj < DATA_BLOCK2; jj += blockDim.x)
                {
                    if ((jj + currentStart) >= data_length) continue;
                    A0temp[jj] = dataConstants[(0 * nChannels + channel) * data_length + currentStart + jj];
                    A1temp[jj] = dataConstants[(1 * nChannels + channel) * data_length + currentStart + jj];
                    B0temp[jj] = dataConstants[(2 * nChannels + channel) * data_length + currentStart + jj];
                    B1temp[jj] = dataConstants[(3 * nChannels + channel) * data_length + currentStart + jj];

                    dataFreqs[jj] = dataFreqsIn[currentStart + jj];

                    //if ((jj + currentStart < 3) && (binNum == 0) & (channel == 0))
                    //    printf("check %e %e, %e %e, %e %e, %e %e, %e \n", A0temp[jj], A1temp[jj], B0temp[jj], B1temp[jj], dataFreqs[jj]);

                }
                __syncthreads();
                if (binNum < numBinAll)
                {
                    for (int jj = 0; jj < DATA_BLOCK2; jj += 1)
                    {
                        if ((jj + currentStart) >= data_length) continue;
                        freq = dataFreqs[jj];
                        trans_complex = templateChannels[((jj + currentStart) * nChannels + channel) * numBinAll + binNum];

                        if ((prevFreq != 0.0) && (jj + currentStart > 0))
                        {
                            A0 = A0temp[jj]; // constants will need to be aligned with 1..n-1 because there are data_length - 1 bins
                            A1 = A1temp[jj];
                            B0 = B0temp[jj];
                            B1 = B1temp[jj];

                            r1 = (trans_complex - prev_trans_complex)/(freq - prevFreq);
                            midFreq = (freq + prevFreq)/2.0;

                            r0 = trans_complex - r1 * (freq - midFreq);

                            //if (((binNum == 767) || (binNum == 768)) & (channel == 0))
                            //    printf("CHECK2: %d %d %d %e %e\n", jj + currentStart, binNum, jj, A0); // , %e %e, %e %e, %e %e, %e %e,  %e %e,  %e %e , %e\n", ind, binNum, jj + currentStart, A0, A1, B0, B1, freq, prevFreq, trans_complex, prev_trans_complex, midFreq);

                            r1Conj = gcmplx::conj(r1);

                            tempLike1 += A0 * gcmplx::conj(r0) + A1 * r1Conj;

                            mag_r0 = gcmplx::abs(r0);
                            tempLike2 += B0 * (mag_r0 * mag_r0) + 2. * B1 * gcmplx::real(r0 * r1Conj);
                        }

                        prev_trans_complex = trans_complex;
                        prevFreq = freq;
                    }
                }
                currentStart += DATA_BLOCK2;
            }
        }
        if (binNum < numBinAll)
        {
            likeOut1[binNum] = tempLike1;
            likeOut2[binNum] = tempLike2;
        }
    }
}
#else
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

                    //if (((binNum == 767) || (binNum == 768)) & (channel == 0))
                    //    printf("CHECK2: %d %d %d %e %e\n", jj + currentStart, binNum, jj, A0); // , %e %e, %e %e, %e %e, %e %e,  %e %e,  %e %e , %e\n", ind, binNum, jj + currentStart, A0, A1, B0, B1, freq, prevFreq, trans_complex, prev_trans_complex, midFreq);

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

CUDA_KERNEL
void noiseweight_template(cmplx* templateChannels, double* noise_weight_times_df, int ind_start, int length, int data_stream_length)
{
    int start, increment;
    #ifdef __CUDACC__
    start = threadIdx.x + blockDim.x * blockIdx.x;
    increment = gridDim.x * blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma parallel omp for
    #endif
    for (int i = start; i < length; i += increment)
    {
        for (int j = 0; j < 3; j+= 1)
        {
            templateChannels[j * length + i] = templateChannels[j * length + i] * noise_weight_times_df[j * data_stream_length + ind_start + i];
        }
    }
}

#define NUM_THREADS_LIKE 256

#ifdef __CUDACC__
void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll)
{

    cudaStream_t streams[numBinAll];
    cublasHandle_t handle;

    cuDoubleComplex result_d_h[numBinAll];
    cuDoubleComplex result_h_h[numBinAll];

    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf ("CUBLAS initialization failed\n");
      exit(0);
    }

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        int nblocks = std::ceil((length_bin_i + NUM_THREADS_LIKE -1)/NUM_THREADS_LIKE);
        cudaStreamCreate(&streams[bin_i]);

        noiseweight_template
        <<<nblocks, NUM_THREADS_LIKE, 0, streams[bin_i]>>>(templateChannels, noise_weight_times_df, ind_start, length_bin_i, data_stream_length);
        cudaStreamSynchronize(streams[bin_i]);

        for (int j = 0; j < 3; j += 1)
        {

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

            d_h[bin_i] += 4.0 * cuCreal(result_d_h[bin_i]);

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


            h_h[bin_i] += 4.0 * cuCreal(result_h_h[bin_i]);

        }
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
    cublasDestroy(handle);

}

#else
void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll)
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
        (templateChannels, noise_weight_times_df, ind_start, length_bin_i, data_stream_length);

        for (int j = 0; j < 3; j += 1)
        {

            cblas_zdotc_sub(length_bin_i,
                              (void*)&dataChannels[j * data_stream_length + ind_start], 1,
                              (void*)&templateChannels[j * length_bin_i], 1,
                              (void*)&result_d_h[bin_i]);

            d_h[bin_i] += 4.0 * result_d_h[bin_i].real();

            cblas_zdotc_sub(length_bin_i,
                              (void*)&templateChannels[j * length_bin_i], 1,
                              (void*)&templateChannels[j * length_bin_i], 1,
                              (void*)&result_h_h[bin_i]);

            h_h[bin_i] += 4.0 * result_h_h[bin_i].real();
            //printf("%e %e\n", cuCreal(result_h_h[bin_i]), cuCreal(result_d_h[bin_i]));

        }
    }
}
#endif
