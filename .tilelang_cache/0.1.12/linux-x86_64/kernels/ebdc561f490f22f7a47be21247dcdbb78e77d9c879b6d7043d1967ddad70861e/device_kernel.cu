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
  bfloat16_t X_local[40];
  float X_pow_local[40];
  float X_powsum[1];
  bfloat16_t W_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    *(uint4*)(((bfloat16_t*)X_shared) + ((i * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(X + ((i * 1024) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 5; ++i_1) {
    *(uint4*)(X_local + (i_1 * 8)) = *(uint4*)(((bfloat16_t*)X_shared) + ((i_1 * 1024) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 10; ++i_2) {
    float4 __1;
    uint2 __2;
      uint2 v_ = *(uint2*)(X_local + (i_2 * 4));
      *(uint1*)(&(__2.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.x)))));
      *(uint1*)(&(__2.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v_.y)))));
    ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__2))[0]);
    ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__2))[1]);
    *(float4*)(X_pow_local + (i_2 * 4)) = __1;
  }
  X_powsum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 40; ++rv) {
    X_powsum[0] = (X_powsum[0] + X_pow_local[(((rv % 5) * 8) + (rv / 5))]);
  }
  __syncthreads();
  X_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(X_powsum[0], (&(((float*)workspace)[0])));
  X_powsum[0] = rsqrtf(((X_powsum[0] / 0x1.4p+12f/*5.120000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_3 = 0; i_3 < 5; ++i_3) {
    *(uint4*)(W_local_cast + 0) = *(uint4*)(W + ((i_3 * 1024) + (((int)threadIdx.x) * 8)));
    for (int vec = 0; vec < 2; ++vec) {
      float broadcast_var = 0x1p+0f/*1.000000e+00*/;
      uint2 __3;
      float4 __4;
        float4 __5;
        uint2 v__1 = *(uint2*)(X_local + ((i_3 * 8) + (vec * 4)));
        ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
        ((float2*)(&__5))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
        float4 __6;
          float4 v__2 = make_float4(X_powsum[0], X_powsum[0], X_powsum[0], X_powsum[0]);
          float4 __7;
            float4 __8;
            uint2 v__3 = *(uint2*)(W_local_cast + (vec * 4));
            ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
            ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
            float4 v__4 = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
            __7.x = (__8.x+v__4.x);
            __7.y = (__8.y+v__4.y);
            __7.z = (__8.z+v__4.z);
            __7.w = (__8.w+v__4.w);
          __6.x = (v__2.x*__7.x);
          __6.y = (v__2.y*__7.y);
          __6.z = (v__2.z*__7.z);
          __6.w = (v__2.w*__7.w);
        __4.x = (__5.x*__6.x);
        __4.y = (__5.y*__6.y);
        __4.z = (__5.z*__6.z);
        __4.w = (__5.w*__6.w);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&__4))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&__4))[1]);
      *(uint2*)(X_local + ((i_3 * 8) + (vec * 4))) = __3;
    }
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 5; ++i_4) {
    *(uint4*)(Y + ((i_4 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(X_local + (i_4 * 8));
  }
}

