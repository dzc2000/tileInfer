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
  bfloat16_t X_local[2];
  float X_pow_local[2];
  float X_powsum[1];
  bfloat16_t W_local_cast[2];
  *(uint1*)(((bfloat16_t*)X_shared) + (((int)threadIdx.x) * 2)) = *(uint1*)(X + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2)));
  *(uint1*)(X_local + 0) = *(uint1*)(((bfloat16_t*)X_shared) + (((int)threadIdx.x) * 2));
  float2 __1;
  uint1 __2;
    uint1 v_ = *(uint1*)(X_local + 0);
    *(uint1*)(&(__2.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.x)))));
  ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__2))[0]);
  *(float2*)(X_pow_local + 0) = __1;
  X_powsum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 2; ++rv) {
    X_powsum[0] = (X_powsum[0] + X_pow_local[rv]);
  }
  __syncthreads();
  X_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(X_powsum[0], (&(((float*)workspace)[0])));
  X_powsum[0] = rsqrtf(((X_powsum[0] / 0x1p+8f/*2.560000e+02*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  *(uint1*)(W_local_cast + 0) = *(uint1*)(W + (((int)threadIdx.x) * 2));
  float broadcast_var = 0x1p+0f/*1.000000e+00*/;
  uint1 __3;
  float2 __4;
    float2 __5;
    uint1 v__1 = *(uint1*)(X_local + 0);
    ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
    float2 __6;
      float2 v__2 = make_float2(X_powsum[0], X_powsum[0]);
      float2 __7;
        float2 __8;
        uint1 v__3 = *(uint1*)(W_local_cast + 0);
        ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
        float2 v__4 = make_float2(broadcast_var, broadcast_var);
        __7.x = (__8.x+v__4.x);
        __7.y = (__8.y+v__4.y);
      __6.x = (v__2.x*__7.x);
      __6.y = (v__2.y*__7.y);
    __4.x = (__5.x*__6.x);
    __4.y = (__5.y*__6.y);
  (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&__4))[0]);
  *(uint1*)(X_local + 0) = __3;
  *(uint1*)(Y + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(X_local + 0);
}

