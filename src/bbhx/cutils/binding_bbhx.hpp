#ifndef __BINDING_BBHX_HPP__
#define __BINDING_BBHX_HPP__

#include "PhenomHMWaveform.hh"
#include "Response.hh"
#include "WaveformBuild.hh"
#include "Likelihood.hh"
#include "Interpolate.hh"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gbt_global.h"

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
template<typename T>
using array_type = cai::cuda_array_t<T>;
#define BBHxComputationWrap BBHxComputationWrapGPU
#else
template<typename T>
using array_type = py::array_t<T>;
#define BBHxComputationWrap BBHxComputationWrapCPU
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define OrbitsWrap_bbhx OrbitsWrapGPU_bbhx
#else
#define OrbitsWrap_bbhx OrbitsWrapCPU_bbhx
#endif


class ReturnPointerBase {
  public:
    template<typename T>
    static T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
        
#else
        py::buffer_info buf1 = input1.request();

        if (buf1.size != N * multiplier)
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

    template<typename T>
    static T* return_pointer(array_type<T> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
#else
        py::buffer_info buf1 = input1.request();
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

    static cmplx* return_pointer_cmplx(array_type<std::complex<double>> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        cmplx *ptr1 = (cmplx *)(input1.get_compatible_typed_pointer());
#else
        py::buffer_info buf1 = input1.request();
        cmplx* ptr1 = static_cast<cmplx *>(buf1.ptr);
#endif
        return ptr1;
    };

};

class OrbitsWrap_bbhx : public ReturnPointerBase{
  public:
    Orbits *orbits;
    OrbitsWrap_bbhx(double dt_, int N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);

        orbits = new Orbits(dt_, N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap_bbhx(){
        delete orbits;
    };
};


class BBHxComputationWrap : public ReturnPointerBase {
  public:
    // OrbitsWrap_bbhx *orbits;
    // BBHxComputationWrap(OrbitsWrap_bbhx *orbits_)
    // {
    //     orbits = orbits_;
    // };
    // ~BBHxComputationWrap{};

    void get_phenomhm_ringdown_frequencies(
        array_type<double>fringdown,
        array_type<double>fdamp,
        array_type<double>m1,
        array_type<double>m2,
        array_type<double>chi1z,
        array_type<double>chi2z,
        array_type<int>ells_in,
        array_type<int>mm_in,
        int numModes,
        int numBinAll
    ){
        get_phenomhm_ringdown_frequencies_wrap(
            return_pointer_and_check_length(fringdown, "fringdown", numModes * numBinAll, 1),
            return_pointer_and_check_length(fdamp, "fdamp", numModes * numBinAll, 1),
            return_pointer_and_check_length(m1, "m1", numBinAll, 1),
            return_pointer_and_check_length(m2, "m2", numBinAll, 1),
            return_pointer_and_check_length(chi1z, "chi1z", numBinAll, 1),
            return_pointer_and_check_length(chi2z, "chi2z", numBinAll, 1),
            return_pointer_and_check_length(ells_in, "ells_in", numModes, 1),
            return_pointer_and_check_length(mm_in, "ells_in", numModes, 1),
            numModes,
            numBinAll
        );
    };

    void get_phenomd_ringdown_frequencies(
        array_type<double>fringdown,
        array_type<double>fdamp,
        array_type<double>m1,
        array_type<double>m2,
        array_type<double>chi1z,
        array_type<double>chi2z,
        int numBinAll,
        array_type<double>y_rd_all,
        array_type<double>c1_rd_all,
        array_type<double>c2_rd_all,
        array_type<double>c3_rd_all,
        array_type<double>y_dm_all,
        array_type<double>c1_dm_all,
        array_type<double>c2_dm_all,
        array_type<double>c3_dm_all,
        double dspin
    ){
        get_phenomd_ringdown_frequencies_wrap(
            return_pointer(fringdown, "fringdown"), // , numBinAll, 1),
            return_pointer(fdamp, "fdamp"),  // , numBinAll, 1),
            return_pointer_and_check_length(m1, "m1", numBinAll, 1),
            return_pointer_and_check_length(m2, "m2", numBinAll, 1),
            return_pointer_and_check_length(chi1z, "chi1z", numBinAll, 1),
            return_pointer_and_check_length(chi2z, "chi2z", numBinAll, 1),
            numBinAll,
            return_pointer(y_rd_all, "y_rd_all"),
            return_pointer(c1_rd_all, "c1_rd_all"),
            return_pointer(c2_rd_all, "c2_rd_all"),
            return_pointer(c3_rd_all, "c3_rd_all"),
            return_pointer(y_dm_all, "y_dm_all"),
            return_pointer(c1_dm_all, "c1_dm_all"),
            return_pointer(c2_dm_all, "c2_dm_all"),
            return_pointer(c3_dm_all, "c3_dm_all"),
            dspin
        );
    };

    
    void waveform_amp_phase_wrap(
        array_type<double>waveformOut,
        array_type<int>ells_in,
        array_type<int>mms_in,
        array_type<double>freqs,               /**< Frequency points at which to evaluate the waveform (Hz) */
        array_type<double>m1_SI,                        /**< mass of companion 1 (kg) */
        array_type<double>m2_SI,                        /**< mass of companion 2 (kg) */
        array_type<double>chi1z,                        /**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
        array_type<double>chi2z,                        /**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
        array_type<double>distance,               /**< distance of source (m) */
        array_type<double>f_ref,                        /**< Reference frequency */
        int numModes,
        int length,
        int numBinAll,
        array_type<double>Mf_RD_lm_all,
        array_type<double>Mf_DM_lm_all,
        int run_phenomd
    ){
        waveform_amp_phase(
            return_pointer(waveformOut, "waveformOut"),
            return_pointer_and_check_length(ells_in, "ells_in", numModes, 1),
            return_pointer_and_check_length(mms_in, "mms_in", numModes, 1),
            return_pointer_and_check_length(freqs, "freqs", length, numBinAll * numModes),
            return_pointer_and_check_length(m1_SI, "m1_SI", numBinAll, 1),
            return_pointer_and_check_length(m2_SI, "m2_SI", numBinAll, 1),
            return_pointer_and_check_length(chi1z, "chi1z", numBinAll, 1),
            return_pointer_and_check_length(chi2z, "chi2z", numBinAll, 1),
            return_pointer_and_check_length(distance, "distance", numBinAll, 1),
            return_pointer_and_check_length(f_ref, "f_ref", numBinAll, 1),
            numModes,
            length,
            numBinAll,
            return_pointer_and_check_length(Mf_RD_lm_all, "Mf_RD_lm_all", numBinAll * (numModes + 1), 1), // + 1 is phenomd 22
            return_pointer_and_check_length(Mf_DM_lm_all, "Mf_DM_lm_all", numBinAll * (numModes + 1), 1), // + 1 is phenomd 22
            run_phenomd
        );
    };

    void LISA_response(
        array_type<double>response_out,
        array_type<int>ells_in,
        array_type<int>mms_in,
        array_type<double>freqs,   /**< Frequency points at which to evaluate the waveform (Hz) */
        array_type<double>phi_ref, /**< reference orbital phase (rad) */
        array_type<double>inc,
        array_type<double>lam,
        array_type<double>beta,
        array_type<double>psi,
        int TDItag, bool rescaled, bool tdi2, int order_fresnel_stencil,
        int numModes,
        int length,
        int numBinAll,
        int includesAmps,
        OrbitsWrap_bbhx *orbits
    ){
        FastLISAResponse fastlisa(orbits->orbits);

        fastlisa.LISA_response(
            return_pointer_and_check_length(response_out, "response_out", length, (8 + includesAmps) * numModes * numBinAll),
            return_pointer_and_check_length(ells_in, "ells_in", numModes, 1),
            return_pointer_and_check_length(mms_in, "mms_in", numModes, 1),
            return_pointer_and_check_length(freqs, "freqs", length, numModes * numBinAll),
            return_pointer_and_check_length(phi_ref, "phi_ref", numBinAll, 1),
            return_pointer_and_check_length(inc, "inc", numBinAll, 1),
            return_pointer_and_check_length(lam, "lam", numBinAll, 1),
            return_pointer_and_check_length(beta, "beta", numBinAll, 1),
            return_pointer_and_check_length(psi, "psi", numBinAll, 1),
            TDItag, rescaled, tdi2, order_fresnel_stencil,
            numModes,
            length,
            numBinAll,
            includesAmps
        );
        
    };

    void InterpTDI_wrap(array_type<long>templateChannels_ptrs, array_type<double>dataFreqs, array_type<double>freqs, array_type<double>propArrays, array_type<double>c1, array_type<double>c2, array_type<double>c3, array_type<double>t_start, array_type<double>t_end, int length, int data_length, int numBinAll, int numModes, array_type<long>inds_ptrs, array_type<int>inds_start, array_type<int>ind_lengths)
    {
        InterpTDI(
            return_pointer_and_check_length(templateChannels_ptrs, "templateChannels_ptrs", numBinAll, 1),
            return_pointer_and_check_length(dataFreqs, "dataFreqs", data_length, 1),
            return_pointer_and_check_length(freqs, "freqs", numBinAll * numModes * length, 1),
            return_pointer_and_check_length(propArrays, "propArrays", numBinAll * numModes * length, 9),
            return_pointer_and_check_length(c1, "c1", numBinAll * numModes * length, 9),
            return_pointer_and_check_length(c2, "c2", numBinAll * numModes * length, 9),
            return_pointer_and_check_length(c3, "c3", numBinAll * numModes * length, 9),
            return_pointer_and_check_length(t_start, "t_start", numBinAll, 1),
            return_pointer_and_check_length(t_end, "t_end", numBinAll, 1),
            length, data_length, numBinAll, numModes, 
            return_pointer_and_check_length(inds_ptrs, "inds_ptrs", numBinAll, 1), 
            return_pointer_and_check_length(inds_start, "inds_start", numBinAll, 1), 
            return_pointer_and_check_length(ind_lengths, "ind_lengths", numBinAll, 1)
        );
    };

    void direct_sum_wrap(array_type<std::complex<double>>templateChannels,
        array_type<double>bbh_buffer,
        int numBinAll, int data_length, int nChannels, int numModes, array_type<double>t_start, array_type<double>t_end)
    {
        direct_sum(
            return_pointer_cmplx(templateChannels, "templateChannels"),
            return_pointer(bbh_buffer, "bbh_buffer"),
            numBinAll, data_length, nChannels, numModes, 
            return_pointer_and_check_length(t_start, "t_start", numBinAll, 1),
            return_pointer_and_check_length(t_end, "t_end", numBinAll, 1)
        );
    };

    void hdyn_wrap(array_type<std::complex<double>>likeOut1, array_type<std::complex<double>>likeOut2, array_type<std::complex<double>>templateChannels, array_type<std::complex<double>>dataConstants, array_type<double>dataFreqs, int numBinAll, int data_length, int nChannels){
        hdyn(
            return_pointer_cmplx(likeOut1, "likeOut1"), 
            return_pointer_cmplx(likeOut2, "likeOut2"), 
            return_pointer_cmplx(templateChannels, "templateChannels"), 
            return_pointer_cmplx(dataConstants, "dataConstants"), 
            return_pointer_and_check_length(dataFreqs, "dataFreqs", data_length, 1),
            numBinAll, data_length, nChannels
        );
    };

    void direct_like_wrap(array_type<std::complex<double>>d_h, array_type<std::complex<double>>h_h, array_type<std::complex<double>>dataChannels, array_type<double>noise_weight_times_df, array_type<long>templateChannels_ptrs, array_type<int>inds_start, array_type<int>ind_lengths, int data_stream_length, int numBinAll){
        direct_like(
            return_pointer_cmplx(d_h, "d_h"), 
            return_pointer_cmplx(h_h, "h_h"), 
            return_pointer_cmplx(dataChannels, "dataChannels"), 
            return_pointer_and_check_length(noise_weight_times_df, "noise_weight_times_df", data_stream_length, 3),
            return_pointer_and_check_length(templateChannels_ptrs, "templateChannels_ptrs", numBinAll, 1),
            return_pointer_and_check_length(inds_start, "inds_start", numBinAll, 1), 
            return_pointer_and_check_length(ind_lengths, "ind_lengths", numBinAll, 1),
            data_stream_length, numBinAll
        );
    };

    void prep_hdyn(array_type<std::complex<double>>A0_in, array_type<std::complex<double>>A1_in, array_type<std::complex<double>>B0_in, array_type<std::complex<double>>B1_in, array_type<std::complex<double>>d_arr, array_type<std::complex<double>>h0_arr, array_type<double>S_n_arr, double df, array_type<int>bins, array_type<double>f_dense, array_type<double>f_m_arr, int data_length, int nchannels, int length_f_rel){
        prep_hdyn_wrap(
            return_pointer_cmplx(A0_in, "A0_in"), 
            return_pointer_cmplx(A1_in, "A1_in"), 
            return_pointer_cmplx(B0_in, "B0_in"), 
            return_pointer_cmplx(B1_in, "B1_in"), 
            return_pointer_cmplx(d_arr, "d_arr"), 
            return_pointer_cmplx(h0_arr, "h0_arr"), 
            return_pointer_and_check_length(S_n_arr, "S_n_arr", data_length * nchannels, 1),
            df, 
            return_pointer_and_check_length(bins, "bins", length_f_rel, 1), 
            return_pointer_and_check_length(f_dense, "f_dense", data_length, 1),
            return_pointer_and_check_length(f_m_arr, "bins", length_f_rel, 1),
            data_length, nchannels, length_f_rel
        );
    };

    void interpolate_wrap(array_type<double>freqs, array_type<double>propArrays,
                 array_type<double>B, array_type<double>upper_diag, array_type<double>diag, array_type<double>lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll){
        interpolate(
            return_pointer_and_check_length(freqs, "freqs", length, numModes * numBinAll),
            return_pointer_and_check_length(propArrays, "propArrays", length, numInterpParams * numModes * numBinAll),
            return_pointer_and_check_length(B, "B", length, numInterpParams * numModes * numBinAll),
            return_pointer_and_check_length(upper_diag, "upper_diag", length, numInterpParams * numModes * numBinAll),
            return_pointer_and_check_length(diag, "diag", length, numInterpParams * numModes * numBinAll),
            return_pointer_and_check_length(lower_diag, "lower_diag", length, numInterpParams * numModes * numBinAll),
            length, numInterpParams, numModes, numBinAll
        );
    };
};

#endif // __BINDING_BBHX_HPP__