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
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* X_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  bfloat16_t X_local[1];
  float X_pow_local[1];
  float X_powsum[1];
  ((bfloat16_t*)X_shared)[((int)threadIdx.x)] = X[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
  X_local[0] = ((bfloat16_t*)X_shared)[((int)threadIdx.x)];
  X_pow_local[0] = ((float)(X_local[0] * X_local[0]));
  X_powsum[0] = 0x0p+0f/*0.000000e+00*/;
  X_powsum[0] = (X_powsum[0] + X_pow_local[0]);
  __syncthreads();
  X_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(X_powsum[0], (&(((float*)workspace)[0])));
  X_powsum[0] = rsqrtf(((X_powsum[0] / 0x1p+7f/*1.280000e+02*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  X_local[0] = ((bfloat16_t)(((float)X_local[0]) * (X_powsum[0] * ((float)W[((int)threadIdx.x)]))));
  Y[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = X_local[0];
}

