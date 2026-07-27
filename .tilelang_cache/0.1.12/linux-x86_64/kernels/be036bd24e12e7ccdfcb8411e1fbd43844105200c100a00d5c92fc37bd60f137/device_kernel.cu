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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y) {
  float acc[1];
  bfloat16_t x_val[1];
  bfloat16_t w_val[1];
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int k = 0; k < 4; ++k) {
    bool is_valid = (0 <= (((int)blockIdx.y) - k));
    x_val[0] = X[(((max((((int)blockIdx.y) - k), 0) * 5120) + (((int)blockIdx.z) * 128)) + ((int)threadIdx.x))];
    w_val[0] = W[((((((int)blockIdx.z) * 512) + (((int)threadIdx.x) * 4)) + 3) - k)];
    float condval;
    if ((0 <= (((int)blockIdx.y) - k))) {
      condval = ((float)x_val[0]);
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    float x_f = condval;
    acc[0] = (acc[0] + (x_f * ((float)w_val[0])));
  }
  acc[0] = (acc[0] * (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - acc[0])))));
  Y[(((((int)blockIdx.y) * 5120) + (((int)blockIdx.z) * 128)) + ((int)threadIdx.x))] = ((bfloat16_t)acc[0]);
}

