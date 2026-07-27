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

extern "C" __global__ void main_kernel(bfloat16_t* __restrict__ NR, const bfloat16_t* __restrict__ R, const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(bfloat16_t* __restrict__ NR, const bfloat16_t* __restrict__ R, const bfloat16_t* __restrict__ W, const bfloat16_t* __restrict__ X, bfloat16_t* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* R_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* X_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  bfloat16_t X_local[40];
  bfloat16_t R_local[40];
  float pow_local[40];
  float pow_sum[1];
  bfloat16_t W_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    *(uint4*)(((bfloat16_t*)X_shared) + ((i * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(X + (((((int)blockIdx.x) * 5120) + (i * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 5; ++i_1) {
    *(uint4*)(X_local + (i_1 * 8)) = *(uint4*)(((bfloat16_t*)X_shared) + ((i_1 * 1024) + (((int)threadIdx.x) * 8)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 5; ++i_2) {
    *(uint4*)(((bfloat16_t*)R_shared) + ((i_2 * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(R + (((((int)blockIdx.x) * 5120) + (i_2 * 1024)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 5; ++i_3) {
    *(uint4*)(R_local + (i_3 * 8)) = *(uint4*)(((bfloat16_t*)R_shared) + ((i_3 * 1024) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 10; ++i_4) {
    float4 __1;
      float4 __2;
      uint2 v_ = *(uint2*)(X_local + (i_4 * 4));
      ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
      ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
      float4 __3;
      uint2 v__1 = *(uint2*)(R_local + (i_4 * 4));
      ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
      ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
      __1.x = (__2.x+__3.x);
      __1.y = (__2.y+__3.y);
      __1.z = (__2.z+__3.z);
      __1.w = (__2.w+__3.w);
    float4 x_f = __1;
    uint2 __4;
    (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&x_f))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&x_f))[1]);
    *(uint2*)(X_local + (i_4 * 4)) = __4;
    float4 __5;
      __5.x = (x_f.x*x_f.x);
      __5.y = (x_f.y*x_f.y);
      __5.z = (x_f.z*x_f.z);
      __5.w = (x_f.w*x_f.w);
    *(float4*)(pow_local + (i_4 * 4)) = __5;
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 5; ++i_5) {
    *(uint4*)(NR + (((((int)blockIdx.x) * 5120) + (i_5 * 1024)) + (((int)threadIdx.x) * 8))) = *(uint4*)(X_local + (i_5 * 8));
  }
  pow_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 40; ++rv) {
    pow_sum[0] = (pow_sum[0] + pow_local[(((rv % 5) * 8) + (rv / 5))]);
  }
  __syncthreads();
  pow_sum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(pow_sum[0], (&(((float*)workspace)[0])));
  pow_sum[0] = rsqrtf(((pow_sum[0] / 0x1.4p+12f/*5.120000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_6 = 0; i_6 < 5; ++i_6) {
    *(uint4*)(W_local_cast + 0) = *(uint4*)(W + ((i_6 * 1024) + (((int)threadIdx.x) * 8)));
    for (int vec = 0; vec < 2; ++vec) {
      float broadcast_var = 0x1p+0f/*1.000000e+00*/;
      uint2 __6;
      float4 __7;
        float4 __8;
        uint2 v__2 = *(uint2*)(X_local + ((i_6 * 8) + (vec * 4)));
        ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[0]);
        ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[1]);
        float4 __9;
          float4 v__3 = make_float4(pow_sum[0], pow_sum[0], pow_sum[0], pow_sum[0]);
          float4 __10;
            float4 __11;
            uint2 v__4 = *(uint2*)(W_local_cast + (vec * 4));
            ((float2*)(&__11))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
            ((float2*)(&__11))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
            float4 v__5 = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
            __10.x = (__11.x+v__5.x);
            __10.y = (__11.y+v__5.y);
            __10.z = (__11.z+v__5.z);
            __10.w = (__11.w+v__5.w);
          __9.x = (v__3.x*__10.x);
          __9.y = (v__3.y*__10.y);
          __9.z = (v__3.z*__10.z);
          __9.w = (v__3.w*__10.w);
        __7.x = (__8.x*__9.x);
        __7.y = (__8.y*__9.y);
        __7.z = (__8.z*__9.z);
        __7.w = (__8.w*__9.w);
      (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&__7))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&__7))[1]);
      *(uint2*)(X_local + ((i_6 * 8) + (vec * 4))) = __6;
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 5; ++i_7) {
    *(uint4*)(Y + (((((int)blockIdx.x) * 5120) + (i_7 * 1024)) + (((int)threadIdx.x) * 8))) = *(uint4*)(X_local + (i_7 * 8));
  }
}

