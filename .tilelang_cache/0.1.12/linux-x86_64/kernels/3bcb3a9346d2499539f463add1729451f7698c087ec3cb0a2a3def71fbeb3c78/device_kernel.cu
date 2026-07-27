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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ COS, const bfloat16_t* __restrict__ SIN, const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ COS, const bfloat16_t* __restrict__ SIN, const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* x_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* w_shared = ((void*)((char*)buf_dyn_shmem + 512));
  void* cos_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* sin_shared = ((void*)((char*)buf_dyn_shmem + 1088));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 1152));
  bfloat16_t x_shared_local_cast[2];
  float x_local[2];
  float pow_local[2];
  float pow_sum[1];
  *(uint1*)(((bfloat16_t*)x_shared) + (((int)threadIdx.x) * 2)) = *(uint1*)(X + ((((int)blockIdx.y) * 256) + (((int)threadIdx.x) * 2)));
  *(uint1*)(((bfloat16_t*)w_shared) + (((int)threadIdx.x) * 2)) = *(uint1*)(W + (((int)threadIdx.x) * 2));
  if (((int)threadIdx.x) < 32) {
    ((bfloat16_t*)cos_shared)[((int)threadIdx.x)] = COS[((int)threadIdx.x)];
    ((bfloat16_t*)sin_shared)[((int)threadIdx.x)] = SIN[((int)threadIdx.x)];
  }
  *(uint1*)(x_shared_local_cast + 0) = *(uint1*)(((bfloat16_t*)x_shared) + (((int)threadIdx.x) * 2));
  float2 __1;
  uint1 v_ = *(uint1*)(x_shared_local_cast + 0);
  ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
  *(float2*)(x_local + 0) = __1;
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    pow_local[i] = (x_local[i] * x_local[i]);
  }
  pow_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 2; ++rv) {
    pow_sum[0] = (pow_sum[0] + pow_local[rv]);
  }
  __syncthreads();
  pow_sum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(pow_sum[0], (&(((float*)workspace)[0])));
  pow_sum[0] = rsqrtf(((pow_sum[0] / 0x1p+8f/*2.560000e+02*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  __syncthreads();
  float broadcast_var = 0x1p+0f/*1.000000e+00*/;
  float2 __2;
    float2 __3;
      float2 v__1 = *(float2*)(x_local + 0);
      float2 v__2 = make_float2(pow_sum[0], pow_sum[0]);
      __3.x = (v__1.x*v__2.x);
      __3.y = (v__1.y*v__2.y);
    float2 __4;
      float2 v__3 = make_float2(broadcast_var, broadcast_var);
      float2 __5;
      uint1 v__4 = *(uint1*)(((bfloat16_t*)w_shared) + (((int)threadIdx.x) * 2));
      ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
      __4.x = (v__3.x+__5.x);
      __4.y = (v__3.y+__5.y);
    __2.x = (__3.x*__4.x);
    __2.y = (__3.y*__4.y);
  *(float2*)(x_local + 0) = __2;
  uint1 __6;
  float2 v__5 = *(float2*)(x_local + 0);
  (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
  *(uint1*)(((bfloat16_t*)x_shared) + (((int)threadIdx.x) * 2)) = __6;
  __syncthreads();
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    float c = ((float)((bfloat16_t*)cos_shared)[i_1]);
    float s = ((float)((bfloat16_t*)sin_shared)[i_1]);
    float val_i = ((float)((bfloat16_t*)x_shared)[i_1]);
    float val_paired = ((float)((bfloat16_t*)x_shared)[(i_1 + 32)]);
    ((bfloat16_t*)x_shared)[i_1] = ((bfloat16_t)((val_i * c) - (val_paired * s)));
    ((bfloat16_t*)x_shared)[(i_1 + 32)] = ((bfloat16_t)((val_paired * c) + (val_i * s)));
  }
  __syncthreads();
  *(uint1*)(Y + ((((int)blockIdx.y) * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(((bfloat16_t*)x_shared) + (((int)threadIdx.x) * 2));
}

