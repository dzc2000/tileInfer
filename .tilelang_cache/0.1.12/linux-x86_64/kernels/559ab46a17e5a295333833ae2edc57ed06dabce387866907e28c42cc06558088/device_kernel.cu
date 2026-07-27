#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ a, bfloat16_t* __restrict__ o, const bfloat16_t* __restrict__ x);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ a, bfloat16_t* __restrict__ o, const bfloat16_t* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_smem = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  float o_reducer[64];
  bfloat16_t a_frag[64];
  bfloat16_t x_frag[1];
  bfloat16_t o_local_cast[8];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(o_reducer + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_smem)[((i_1 * 1024) + (((int)threadIdx.x) * 8))])), (&(a[((((((int)blockIdx.x) * 196608) + (i_1 * 24576)) + ((((int)threadIdx.x) >> 4) * 3072)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_smem)[(((i_2 * 1024) + (((int)threadIdx.x) * 8)) + 8192)])), (&(a[(((((((int)blockIdx.x) * 196608) + (i_2 * 24576)) + ((((int)threadIdx.x) >> 4) * 3072)) + ((((int)threadIdx.x) & 15) * 8)) + 128)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)a_smem)[(((i_3 * 1024) + (((int)threadIdx.x) * 8)) + 16384)])), (&(a[(((((((int)blockIdx.x) * 196608) + (i_3 * 24576)) + ((((int)threadIdx.x) >> 4) * 3072)) + ((((int)threadIdx.x) & 15) * 8)) + 256)])));
  }
  tl::cp_async_commit();
  for (int i0_n = 0; i0_n < 21; ++i0_n) {
    tl::cp_async_wait<2>();
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      a_frag[i_4] = ((bfloat16_t*)a_smem)[((((i0_n % 3) * 8192) + (i_4 * 128)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)a_smem)[((((i0_n % 3) * 8192) + (i_5 * 1024)) + (((int)threadIdx.x) * 8))])), (&(a[((((((((int)blockIdx.x) * 196608) + (i_5 * 24576)) + ((((int)threadIdx.x) >> 4) * 3072)) + (i0_n * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 384)])));
    }
    tl::cp_async_commit();
    x_frag[0] = x[((i0_n * 128) + ((int)threadIdx.x))];
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      o_reducer[i_6] = (o_reducer[i_6] + (((float)a_frag[i_6]) * ((float)x_frag[0])));
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    a_frag[i_7] = ((bfloat16_t*)a_smem)[((i_7 * 128) + ((int)threadIdx.x))];
  }
  x_frag[0] = x[(((int)threadIdx.x) + 2688)];
  #pragma unroll
  for (int i_8 = 0; i_8 < 64; ++i_8) {
    o_reducer[i_8] = (o_reducer[i_8] + (((float)a_frag[i_8]) * ((float)x_frag[0])));
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 64; ++i_9) {
    a_frag[i_9] = ((bfloat16_t*)a_smem)[(((i_9 * 128) + ((int)threadIdx.x)) + 8192)];
  }
  x_frag[0] = x[(((int)threadIdx.x) + 2816)];
  #pragma unroll
  for (int i_10 = 0; i_10 < 64; ++i_10) {
    o_reducer[i_10] = (o_reducer[i_10] + (((float)a_frag[i_10]) * ((float)x_frag[0])));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 64; ++i_11) {
    a_frag[i_11] = ((bfloat16_t*)a_smem)[(((i_11 * 128) + ((int)threadIdx.x)) + 16384)];
  }
  x_frag[0] = x[(((int)threadIdx.x) + 2944)];
  #pragma unroll
  for (int i_12 = 0; i_12 < 64; ++i_12) {
    o_reducer[i_12] = (o_reducer[i_12] + (((float)a_frag[i_12]) * ((float)x_frag[0])));
  }
  __syncthreads();
  for (int __finred_0 = 0; __finred_0 < 64; ++__finred_0) {
    o_reducer[__finred_0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(o_reducer[__finred_0], (&(((float*)workspace)[0])));
  }
  if (((int)threadIdx.x) == 0) {
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __1;
        float4 v_ = *(float4*)(o_reducer + ((i_13 * 8) + (vec * 4)));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
        *(uint2*)(o_local_cast + (vec * 4)) = __1;
      }
      *(uint4*)(o + ((((int)blockIdx.x) * 64) + (i_13 * 8))) = *(uint4*)(o_local_cast + 0);
    }
  }
}

