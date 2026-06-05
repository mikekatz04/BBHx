// Phase 3L.8 (2026-06-04): GBT's Interpolate.cu provides the LAPACKE
// tridiagonal cubic-spline solver (`fit_cubic_spline_thomas` +
// `fit_cubic_spline_thomas_run`) that the chunked-heterodyne pipeline in
// `lat_chunked_het_kernels.hh` relies on for orbit-spline + signal-spline
// construction. cgbgpu (Phase 3L.7g) pulls the whole `Interpolate.cu` in
// directly because GBGPU has no local Interpolate -- no symbol collisions.
//
// BBHx is different: its local `Interpolate.cu` defines six low-level
// helpers (`prep_splines`, `fill_B`, `interpolate_kern`,
// `fill_coefficients`, `set_spline_constants`, `interpolate`) with the
// same names but different signatures than GBT's. Pulling GBT's
// `Interpolate.cu` in unmodified would cause linker collisions.
//
// This wrapper TU renames the colliding helpers to LAT-prefixed names via
// `#define`, then `#include`s GBT's `Interpolate.cu` as plain text. The
// outward-facing entry points (`fit_cubic_spline_thomas` +
// `fit_cubic_spline_thomas_run`) keep their global names so consumers of
// GBT's `Interpolate.hh` (notably `lat_chunked_het_kernels.hh`) link
// cleanly without modification.
//
// Same source file serves CPU and GPU builds: copied to .cxx for CPU,
// compiled as .cu by nvcc for GPU.

#define prep_splines           bbhx_lat_prep_splines
#define fill_B                 bbhx_lat_fill_B
#define interpolate_kern       bbhx_lat_interpolate_kern
#define fill_coefficients      bbhx_lat_fill_coefficients
#define set_spline_constants   bbhx_lat_set_spline_constants
#define interpolate            bbhx_lat_interpolate
#define eval_kernel            bbhx_lat_eval_kernel
#define eval_wrap              bbhx_lat_eval_wrap

#include "gbt_Interpolate.cu"  // copied from GBT_CUTILS by the CMake rule

#undef prep_splines
#undef fill_B
#undef interpolate_kern
#undef fill_coefficients
#undef set_spline_constants
#undef interpolate
#undef eval_kernel
#undef eval_wrap
