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
  bfloat16_t A_register[8];
  bfloat16_t B_register[8];
  bfloat16_t C_register[8];
  *(uint4*)(A_register + 0) = *(uint4*)(A + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  *(uint4*)(B_register + 0) = *(uint4*)(B + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    C_register[i] = ((A_register[i] * (bfloat16_t(0x1p+0f/*1.000000e+00*/) / (bfloat16_t(0x1p+0f/*1.000000e+00*/) + ((bfloat16_t)expf(((float)(bfloat16_t(0x0p+0f/*0.000000e+00*/) - A_register[i]))))))) * B_register[i]);
  }
  *(uint4*)(C + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(C_register + 0);
}

