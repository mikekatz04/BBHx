#ifndef __INTERPOLATE_H_
#define __INTERPOLATE_H_

class Interpolate{
    double *w;
    double *d_mat;

    double *dl;
    double *d;
    double *du;
    int m;
    int n;
    int to_gpu;

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();

    void prep(double *b_mat, int m_, int n_, int to_gpu_);

    ~Interpolate(); //destructor

    void fit_constants(double *b_mat);
};

void host_fill_B_wave(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_fill_B_response(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_set_spline_constants_wave(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_set_spline_constants_response(ModeContainer *mode_vals, double *b_mat, int f_length, int num_modes);
void host_interpolate(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals, int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv);

#endif //__INTERPOLATE_H_
