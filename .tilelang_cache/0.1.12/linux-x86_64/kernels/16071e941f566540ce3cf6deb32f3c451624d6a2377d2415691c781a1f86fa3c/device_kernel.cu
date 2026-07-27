#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  float C_accum[1];
  bfloat16_t A_local[8];
  bfloat16_t B_local[8];
  float C_reduced[1];
  C_accum[0] = 0x0p+0f/*0.000000e+00*/;
  for (int bk = 0; bk < 12; ++bk) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((bk * 256) + (((int)threadIdx.x) * 8)));
    *(uint4*)(B_local + 0) = *(uint4*)(B + ((((((int)blockIdx.x) * 24576) + (((int)threadIdx.y) * 3072)) + (bk * 256)) + (((int)threadIdx.x) * 8)));
    for (int k = 0; k < 8; ++k) {
      C_accum[0] = (C_accum[0] + (((float)A_local[k]) * ((float)B_local[k])));
    }
  }
  float red_buf0[1];
  float t0[1];
  uint mask[1];
  red_buf0[0] = C_accum[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  C[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] = ((bfloat16_t)red_buf0[0]);
}

