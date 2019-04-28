/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
//#include <reduction.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"
#include "cublas_v2.h"
#include "interpolate.cu"
#include "fdresponse.h"
#include "createGPUHolders.cu"
#include "kernel_response.cu"


using namespace std;

GPUPhenomHM::GPUPhenomHM (int max_length_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    int to_gpu_,
    int to_interp_,
    std::complex<double> *data_stream_, int data_stream_length_){

    max_length = max_length_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    to_gpu = to_gpu_;
    to_interp = to_interp_;
    data_stream = data_stream_;
    data_stream_length = data_stream_length_;

    cudaError_t err;

    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

    hI = new std::complex<double>[data_stream_length*num_modes];
    hII = new std::complex<double>[data_stream_length*num_modes];

    X = new std::complex<double>[data_stream_length*num_modes];
    Y = new std::complex<double>[data_stream_length*num_modes];
    Z = new std::complex<double>[data_stream_length*num_modes];


  mode_vals = cpu_create_modes(num_modes, l_vals, m_vals, max_length, to_gpu, to_interp);

  if (to_gpu == 1){

      gpuErrchk(cudaMalloc(&d_H, 9*num_modes*sizeof(cuDoubleComplex)));

      cuDoubleComplex * ones = new cuDoubleComplex[num_modes];
      for (int i=0; i<(num_modes); i++) ones[i] = make_cuDoubleComplex(1.0, 0.0);
      gpuErrchk(cudaMalloc(&d_ones, num_modes*sizeof(cuDoubleComplex)));
      gpuErrchk(cudaMemcpy(d_ones, ones, num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
      delete ones;

      gpuErrchk(cudaMalloc(&d_hI, data_stream_length*num_modes*sizeof(cuDoubleComplex)));
      gpuErrchk(cudaMalloc(&d_hII, data_stream_length*num_modes*sizeof(cuDoubleComplex)));

      gpuErrchk(cudaMalloc(&d_hI_out, data_stream_length*sizeof(cuDoubleComplex)));
      gpuErrchk(cudaMalloc(&d_hII_out, data_stream_length*sizeof(cuDoubleComplex)));

      gpuErrchk(cudaMalloc(&d_X, data_stream_length*num_modes*sizeof(cuDoubleComplex)));
      gpuErrchk(cudaMalloc(&d_Y, data_stream_length*num_modes*sizeof(cuDoubleComplex)));
      gpuErrchk(cudaMalloc(&d_Z, data_stream_length*num_modes*sizeof(cuDoubleComplex)));

      d_mode_vals = gpu_create_modes(num_modes, l_vals, m_vals, max_length, to_gpu, to_interp);

      gpuErrchk(cudaMalloc(&d_freqs, max_length*sizeof(double)));

      gpuErrchk(cudaMalloc(&d_data_stream, data_stream_length*sizeof(cuDoubleComplex)));
      gpuErrchk(cudaMemcpy(d_data_stream, data_stream, data_stream_length*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

      //gpuErrchk(cudaMalloc(&d_mode_vals, num_modes*sizeof(d_mode_vals)));
      //gpuErrchk(cudaMemcpy(d_mode_vals, mode_vals, num_modes*sizeof(d_mode_vals), cudaMemcpyHostToDevice));

      // DECLARE ALL THE  NECESSARY STRUCTS
      gpuErrchk(cudaMalloc(&d_pHM_trans, sizeof(PhenomHMStorage)));

      gpuErrchk(cudaMalloc(&d_pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients)));

      gpuErrchk(cudaMalloc(&d_amp_prefactors_trans, sizeof(AmpInsPrefactors)));

      gpuErrchk(cudaMalloc(&d_pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp)));

      gpuErrchk(cudaMalloc((void**) &d_q_all_trans, num_modes*sizeof(HMPhasePreComp)));


      double cShift[7] = {0.0,
                           PI_2 /* i shift */,
                           0.0,
                           -PI_2 /* -i shift */,
                           PI /* 1 shift */,
                           PI_2 /* -1 shift */,
                           0.0};

      gpuErrchk(cudaMalloc(&d_cShift, 7*sizeof(double)));

      gpuErrchk(cudaMemcpy(d_cShift, &cShift, 7*sizeof(double), cudaMemcpyHostToDevice));


      // for likelihood
      // --------------
      gpuErrchk(cudaMallocHost((cuDoubleComplex**) &result, sizeof(cuDoubleComplex)));

      stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }
      // ----------------
  }
  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;
}


void GPUPhenomHM::add_interp(int max_interp_length_){
    max_interp_length = max_interp_length_;

    assert(to_interp == 1);
    if (to_gpu == 0){
        out_mode_vals = cpu_create_modes(num_modes, m_vals, l_vals, max_interp_length, to_gpu, 0);
        B = new double[7*max_interp_length*num_modes];
    }
    if (to_gpu){

        h_indices = new int[max_interp_length];
        cudaMalloc(&d_indices, max_interp_length*sizeof(int));
        //d_out_mode_vals = gpu_create_modes(num_modes, m_vals, l_vals, max_interp_length, to_gpu, 0);
        //h_B = new double[2*f_length*num_modes];
        //h_B1 = new double[2*f_length*num_modes];*/
        gpuErrchk(cudaMalloc(&d_B, 7*max_interp_length_*num_modes*sizeof(double)));
    }
}



void GPUPhenomHM::gpu_gen_PhenomHM(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    assert((to_gpu == 1) || (to_gpu == 2));

    GPUPhenomHM::cpu_gen_PhenomHM(freqs_, f_length_,
        m1_, //solar masses
        m2_, //solar masses
        chi1z_,
        chi2z_,
        distance_,
        inclination_,
        phiRef_,
        deltaF_,
        f_ref_);

    // Initialize inputs
    //gpuErrchk(cudaMemcpy(d_mode_vals, mode_vals, num_modes*sizeof(ModeContainer), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_freqs, freqs, f_length*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pHM_trans, pHM_trans, sizeof(PhenomHMStorage), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pAmp_trans, pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_amp_prefactors_trans, amp_prefactors_trans, sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans, pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_q_all_trans, q_all_trans, num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice));

    double M_tot_sec = (m1+m2)*MTSUN_SI;
    /* main: evaluate model at given frequencies */
    NUM_THREADS = 256;
    num_blocks = std::ceil((f_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_modes, num_blocks);
    //printf("blocks %d\n", num_blocks);
    kernel_calculate_all_modes<<<gridDim, NUM_THREADS>>>(d_mode_vals,
          d_pHM_trans,
          d_freqs,
          M_tot_sec,
          d_pAmp_trans,
          d_amp_prefactors_trans,
          d_pDPreComp_all_trans,
          d_q_all_trans,
          amp0,
          num_modes,
          t0,
          phi0,
          d_cShift
      );
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

}


void GPUPhenomHM::cpu_gen_PhenomHM(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    freqs = freqs_;
    f_length = f_length_;
    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    inclination = inclination_;
    phiRef = phiRef_;
    deltaF = deltaF_;
    f_ref = f_ref_;

    for (int i=0; i<num_modes; i++){
        mode_vals[i].length = f_length;
    }

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        f_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        inclination,
        phiRef,
        deltaF,
        f_ref,
        num_modes,
        to_gpu,
        pHM_trans,
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in

}


__global__ void read_out_kernel2(ModeContainer *mode_vals, double *coef0, double *coef1, double *coef2, double *coef3, int mode_i, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    coef0[i] = mode_vals[mode_i].amp[i];
    coef1[i] = mode_vals[mode_i].amp_coeff_1[i];
    coef2[i] = mode_vals[mode_i].amp_coeff_2[i];
    coef3[i] = mode_vals[mode_i].amp_coeff_3[i];
    //phase[i] = mode_vals[mode_i].phase[i];
}

__global__ void debug(ModeContainer *mode_vals, int num_modes, int length){
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.x;
    if (mode_i >= num_modes) return;
    if (i >= length) return;
    double amp = mode_vals[mode_i].amp[i];
    double phase = mode_vals[mode_i].phase[i];
    //phase[i] = mode_vals[mode_i].phase[i];
}

void GPUPhenomHM::gpu_setup_interp_wave(){

    dim3 check_dim(num_modes, num_blocks);
    int check_num_threads = 256;
    /*debug<<<check_dim, NUM_THREADS>>>(d_mode_vals, num_modes, f_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());*/

    fill_B_wave<<<check_dim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, f_length, 2*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_wave<<<check_dim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void GPUPhenomHM::cpu_setup_interp_wave(){
    host_fill_B_wave(mode_vals, B, f_length, num_modes);
    interp.prep(B, f_length, 2*num_modes, 0);
    host_set_spline_constants_wave(mode_vals, B, f_length, num_modes);
}

//void GPUPhenomHM::cpu_setup_interp_response(double f_min, double df, int length_new)


void GPUPhenomHM::gpu_LISAresponseFD(double inc_, double lam_, double beta_, double psi_, double tc_, double tShift_, int TDItag_){
    inc = inc_;
    lam = lam_;
    beta = beta_;
    psi = psi_;
    tc = tc_;
    tShift = tShift_;
    TDItag = TDItag_;

    H = prep_H_info(l_vals, m_vals, num_modes, inc, lam, beta, psi, phi0);
    gpuErrchk(cudaMemcpy(d_H, H, 9*num_modes*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);

    int num_blocks = std::ceil((f_length + NUM_THREADS - 1)/NUM_THREADS);
    dim3 gridDim(num_modes, num_blocks);

    kernel_JustLISAFDresponseTDI_wrap<<<gridDim, NUM_THREADS>>>(d_mode_vals, d_H, d_freqs, d_freqs, d_log10f, d_l_vals, d_m_vals, num_modes, f_length, inc, lam, beta, psi, phi0, tc, tShift, TDItag, 0);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void GPUPhenomHM::cpu_LISAresponseFD(double inc_, double lam_, double beta_, double psi_, double tc_, double tShift_, int TDItag_){
    inc = inc_;
    lam = lam_;
    beta = beta_;
    psi = psi_;
    tc = tc_;
    tShift = tShift_;
    TDItag = TDItag_;

    H = prep_H_info(l_vals, m_vals, num_modes, inc, lam, beta, psi, phi0);
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);
    JustLISAFDresponseTDI_wrap(mode_vals, H, freqs, freqs, d_log10f, l_vals, m_vals, num_modes, f_length, inc, lam, beta, psi, phi0, tc, tShift, TDItag, 0);
}

void GPUPhenomHM::gpu_setup_interp_response(){

    dim3 check_dim(num_modes, num_blocks);
    int check_num_threads = 256;
    /*debug<<<check_dim, NUM_THREADS>>>(d_mode_vals, num_modes, f_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());*/

    fill_B_response<<<check_dim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interp.prep(d_B, f_length, 7*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    set_spline_constants_response<<<check_dim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void GPUPhenomHM::cpu_setup_interp_response(){
    host_fill_B_response(mode_vals, B, f_length, num_modes);
    interp.prep(B, f_length, 7*num_modes, 0);
    host_set_spline_constants_response(mode_vals, B, f_length, num_modes);
}

void GPUPhenomHM::gpu_perform_interp(double f_min, double df, int length_new){
    int num_block_interp = std::ceil((length_new + NUM_THREADS - 1)/NUM_THREADS);
    dim3 interp_dim(num_modes, num_block_interp);
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);
    //printf("NUM MODES %d\n", num_modes);
    interpolate<<<interp_dim, NUM_THREADS>>>(d_X, d_Y, d_Z, d_mode_vals, num_modes, f_min, df, d_log10f, d_freqs, length_new, tc, tShift);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    //TODO need to make this more adaptable (especially for smaller amounts)
}

void GPUPhenomHM::cpu_perform_interp(double f_min, double df, int length_new){
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);
    host_interpolate(X, Y, Z, mode_vals, num_modes, f_min, df, d_log10f, freqs, length_new, tc, tShift);
}

/*
__global__ void debug2(cuDoubleComplex *hI, cuDoubleComplex *hI_out, cuDoubleComplex *ones, int length, int num_modes){
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.x;
    if (mode_i >= num_modes) return;
    if (i >= length) return;
    int j = 0;
    //phase[i] = mode_vals[mode_i].phase[i];
}*/

int GpuVec(cuDoubleComplex* d_A, cuDoubleComplex* d_x, cuDoubleComplex* d_y, const int row,const int col){
cudaError_t cudastat;
cublasStatus_t stat;
int size=row*col;
cublasHandle_t handle;
/*cuDoubleComplex* d_A;  //device matrix
cuDoubleComplex* d_x;  //device vector
cuDoubleComplex* d_y;  //device result
cudastat=cudaMalloc((void**)&d_A,size*sizeof(cuDoubleComplex));
cudastat=cudaMalloc((void**)&d_x,col*sizeof(cuDoubleComplex));
cudastat=cudaMalloc((void**)&d_y,row*sizeof(cuDoubleComplex));// when I copy y to d_y ,can I cout d_y?

cudaMemcpy(d_A,A,sizeof(cuDoubleComplex)*size,cudaMemcpyHostToDevice);  //copy A to device d_A
cudaMemcpy(d_x,x,sizeof(cuDoubleComplex)*col,cudaMemcpyHostToDevice);*/   //copy x to device d_x

cuDoubleComplex alf=make_cuDoubleComplex(1.0,0.0);
cuDoubleComplex beta=make_cuDoubleComplex(0.0,0.0);
    stat=cublasCreate(&handle);
/*int NUM_THREADS = 256;
int num_blockshere = (int)(row + NUM_THREADS -1)/NUM_THREADS;
dim3 likeDim(col, num_blockshere);
debug2<<<likeDim, NUM_THREADS>>>(d_A, d_y, d_x, row, col);
cudaDeviceSynchronize();
gpuErrchk(cudaGetLastError());*/
stat=cublasZgemv(handle,CUBLAS_OP_T,col,row,&alf,d_A,col,d_x,1,&beta,d_y,1);//swap col and row
/*cudaMemcpy(y,d_y,sizeof(cuDoubleComplex)*row,cudaMemcpyDeviceToHost); // copy device result to host
cudaFree(d_A);
cudaFree(d_x);
cudaFree(d_y);*/
cublasDestroy(handle);
return 0;
}


double GPUPhenomHM::Likelihood (int like_length){

     cuDoubleComplex res_out = make_cuDoubleComplex(0.0, 0.0);
     char * status;
     for (int mode_i=0; mode_i<num_modes; mode_i++){
         stat = cublasZdotc(handle, like_length,
                 //d_hI_out, 1,
                 &d_hI[mode_i*like_length], 1,
                 d_data_stream, 1,
                 result);
         status = _cudaGetErrorEnum(stat);
          cudaDeviceSynchronize();
          //printf ("%s\n", status);
          if (stat != CUBLAS_STATUS_SUCCESS) {
                  exit(0);
              }
         res_out = cuCadd(res_out, result[0]);
     }

    //gpuErrchk(cudaGetLastError());


    //return cuCreal(result[0]);
    return cuCreal(res_out);
    //return 0.0;
}

void GPUPhenomHM::Get_Waveform (std::complex<double>* X_, std::complex<double>* Y_, std::complex<double>* Z_) {
  assert(to_gpu == 0);
  memcpy(X_, X, max_interp_length*num_modes*sizeof(std::complex<double>));
  memcpy(Y_, Y, max_interp_length*num_modes*sizeof(std::complex<double>));
  memcpy(Z_, Z, max_interp_length*num_modes*sizeof(std::complex<double>));
}

__global__ void read_out_kernel(ModeContainer *mode_vals, double *amp, double *phase, int mode_i, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    amp[i] = mode_vals[mode_i].amp[i];
    phase[i] = mode_vals[mode_i].phase[i];
}

void GPUPhenomHM::gpu_Get_Waveform (std::complex<double>* hI_) {
  assert(to_gpu == 1);
  gpuErrchk(cudaMemcpy(hI_, d_hI, max_interp_length*num_modes*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));
}

GPUPhenomHM::~GPUPhenomHM() {
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete pDPreComp_all_trans;
  delete q_all_trans;
  cpu_destroy_modes(mode_vals);
  delete hI;
  delete hII;
  delete X;
  delete Y;
  delete Z;

  if (to_gpu == 1){
      cudaFree(d_ones);
      cudaFree(d_hI);
      cudaFree(d_hII);
      cudaFree(d_hI_out);
      cudaFree(d_hII_out);
      cudaFree(d_freqs);
      cudaFree(d_data_stream);
      gpu_destroy_modes(d_mode_vals);
      cudaFree(d_pHM_trans);
      cudaFree(d_pAmp_trans);
      cudaFree(d_amp_prefactors_trans);
      cudaFree(d_pDPreComp_all_trans);
      cudaFree(d_q_all_trans);
      cudaFree(d_cShift);
      cudaFree(result);
      cudaFree(d_X);
      cudaFree(d_Y);
      cudaFree(d_Z);
      cublasDestroy(handle);
  }
  if (to_interp == 1){
      if (to_gpu == 0){
          delete B;
          cpu_destroy_modes(out_mode_vals);
      }
      delete h_indices;
      cudaFree(d_indices);
      cudaFree(d_B);
      gpu_destroy_modes(d_out_mode_vals);
      //delete interp;
  }
}
