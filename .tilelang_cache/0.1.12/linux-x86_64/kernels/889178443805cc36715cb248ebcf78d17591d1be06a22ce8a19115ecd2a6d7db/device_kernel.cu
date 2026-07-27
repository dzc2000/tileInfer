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

extern "C" __global__ void main_kernel(bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(bfloat16_t* __restrict__ S, const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y) {
  float x_local[1];
  float acc[1];
  bfloat16_t s_local[1];
  float w_local[1];
  x_local[0] = ((float)X[(((((int)blockIdx.x) * 5120) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))]);
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int i = 0; i < 4; ++i) {
    if (i < 3) {
      s_local[0] = S[(((((((int)blockIdx.x) * 20480) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.x) * 4)) + i) + 1)];
    } else {
      s_local[0] = ((bfloat16_t)x_local[0]);
    }
    S[((((((int)blockIdx.x) * 20480) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.x) * 4)) + i)] = s_local[0];
    w_local[0] = ((float)W[(((((int)blockIdx.y) * 512) + (((int)threadIdx.x) * 4)) + i)]);
    acc[0] = (acc[0] + (((float)s_local[0]) * w_local[0]));
  }
  acc[0] = (acc[0] * (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - acc[0])))));
  Y[(((((int)blockIdx.x) * 5120) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))] = ((bfloat16_t)acc[0]);
}

