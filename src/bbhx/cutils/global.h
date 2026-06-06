#ifndef __BBHX_GLOBAL_H__
#define __BBHX_GLOBAL_H__

// BBHx's local `global.h` used to duplicate GPUBackendTools' gbt_global.h
// almost verbatim (cuda_complex include, cmplx typedef, CUDA_KERNEL /
// CUDA_SHARED / CUDA_SYNC_THREADS / CUDA_CALLABLE_MEMBER macros, gpuErrchk).
// Phase 3.dedup-followup (2026-06-05) collapsed all of that into a single
// sprint-wide source: pull in GBT's header so BBHx and LAT share one copy.
// The cbbhx CMake targets put ${GBT_CUTILS} on the include path, so this
// just resolves to gpubackendtools/cutils/gbt_global.h.
//
// Semantic deltas vs. BBHx's old global.h, all strict supersets that keep
// existing kernels valid:
//   - CUDA_CALLABLE_MEMBER is `__host__ __device__` (was `__device__`).
//   - THREAD_ZERO checks all three thread axes (was just .x).
//   - GBT also defines CUDA_DEVICE + THREAD_START_X/Y/Z + BLOCK_INCR_X/Y/Z +
//     GRID_INCR_X/Y/Z + BLOCK_START_X/Y/Z. Not used by BBHx kernels today,
//     but harmless.
#include "gbt_global.h"

#endif // __BBHX_GLOBAL_H__
