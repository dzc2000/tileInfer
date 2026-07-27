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

extern "C" __global__ void kernel_kernel(const float* __restrict__ Beta, float* __restrict__ FinalState, const float* __restrict__ G, const float* __restrict__ InitState, const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const float* __restrict__ Beta, float* __restrict__ FinalState, const float* __restrict__ G, const float* __restrict__ InitState, const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* cum_g = ((void*)((char*)buf_dyn_shmem + 0));
  void* S = ((void*)((char*)buf_dyn_shmem + 128));
  void* K_buf = ((void*)((char*)buf_dyn_shmem + 65664));
  void* Q_buf = ((void*)((char*)buf_dyn_shmem + 73856));
  void* V_buf = ((void*)((char*)buf_dyn_shmem + 82048));
  void* beta_buf = ((void*)((char*)buf_dyn_shmem + 90240));
  void* O_buf = ((void*)((char*)buf_dyn_shmem + 90368));
  void* A = ((void*)((char*)buf_dyn_shmem + 106752));
  float acc[1];
  float acc_1[1];
  float acc_2[1];
  float O_buf_local_cast_1[4];
  bfloat16_t Output_local_cast[4];
  float acc_3[1];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float4*)(((float*)S) + ((i * 1024) + (((int)threadIdx.x) * 4))) = *(float4*)(InitState + (((((int)blockIdx.x) * 16384) + (i * 1024)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(uint4*)(((bfloat16_t*)Q_buf) + ((i_1 * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(Q + ((((i_1 * 16384) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) / 3) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    *(uint4*)(((bfloat16_t*)K_buf) + ((i_1 * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(K + ((((i_1 * 16384) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) / 3) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(uint4*)(((bfloat16_t*)V_buf) + ((i_2 * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(V + ((((i_2 * 49152) + ((((int)threadIdx.x) >> 4) * 3072)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  if (((int)threadIdx.x) < 32) {
    ((float*)beta_buf)[((int)threadIdx.x)] = Beta[((((int)threadIdx.x) * 24) + ((int)blockIdx.x))];
  }
  ((float*)cum_g)[0] = G[((int)blockIdx.x)];
  __syncthreads();
  for (int i_3 = 0; i_3 < 31; ++i_3) {
    ((float*)cum_g)[(i_3 + 1)] = (((float*)cum_g)[i_3] + G[(((i_3 * 24) + ((int)blockIdx.x)) + 24)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    acc[0] = 0x0p+0f/*0.000000e+00*/;
    for (int d = 0; d < 128; ++d) {
      acc[0] = (acc[0] + (((float)((bfloat16_t*)Q_buf)[(((i_4 * 256) + ((((int)threadIdx.x) >> 7) * 128)) + d)]) * ((float*)S)[((d * 128) + (((int)threadIdx.x) & 127))]));
    }
    ((float*)O_buf)[((i_4 * 256) + ((int)threadIdx.x))] = (expf(((float*)cum_g)[((i_4 * 2) + (((int)threadIdx.x) >> 7))]) * acc[0]);
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    acc_1[0] = 0x0p+0f/*0.000000e+00*/;
    for (int d_1 = 0; d_1 < 128; ++d_1) {
      acc_1[0] = (acc_1[0] + (((float)((bfloat16_t*)Q_buf)[(((i_5 * 1024) + ((((int)threadIdx.x) >> 5) * 128)) + d_1)]) * ((float)((bfloat16_t*)K_buf)[(((((int)threadIdx.x) & 31) * 128) + d_1)])));
    }
    float condval;
    if (((((int)threadIdx.x) & 31) <= ((i_5 * 8) + (((int)threadIdx.x) >> 5)))) {
      condval = ((expf((((float*)cum_g)[((i_5 * 8) + (((int)threadIdx.x) >> 5))] - ((float*)cum_g)[(((int)threadIdx.x) & 31)])) * ((float*)beta_buf)[(((int)threadIdx.x) & 31)]) * acc_1[0]);
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    ((float*)A)[((i_5 * 256) + ((int)threadIdx.x))] = condval;
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    acc_2[0] = 0x0p+0f/*0.000000e+00*/;
    for (int j = 0; j < 32; ++j) {
      acc_2[0] = (acc_2[0] + (((float*)A)[(((i_6 * 64) + ((((int)threadIdx.x) >> 7) * 32)) + j)] * ((float)((bfloat16_t*)V_buf)[((j * 128) + (((int)threadIdx.x) & 127))])));
    }
    ((float*)O_buf)[((i_6 * 256) + ((int)threadIdx.x))] = (((float*)O_buf)[((i_6 * 256) + ((int)threadIdx.x))] + acc_2[0]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    *(float4*)(O_buf_local_cast_1 + 0) = *(float4*)(((float*)O_buf) + ((i_7 * 1024) + (((int)threadIdx.x) * 4)));
    uint2 __1;
    float4 v_ = *(float4*)(O_buf_local_cast_1 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
    *(uint2*)(Output_local_cast + 0) = __1;
    *(uint2*)(Output + ((((i_7 * 24576) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(uint2*)(Output_local_cast + 0);
  }
  float chunk_g = ((float*)cum_g)[31];
  float chunk_decay = expf(chunk_g);
  #pragma unroll
  for (int i_8 = 0; i_8 < 64; ++i_8) {
    acc_3[0] = 0x0p+0f/*0.000000e+00*/;
    for (int j_1 = 0; j_1 < 32; ++j_1) {
      float w = (expf((chunk_g - ((float*)cum_g)[j_1])) * ((float*)beta_buf)[j_1]);
      acc_3[0] = (acc_3[0] + ((((float)((bfloat16_t*)K_buf)[(((j_1 * 128) + (i_8 * 2)) + (((int)threadIdx.x) >> 7))]) * w) * ((float)((bfloat16_t*)V_buf)[((j_1 * 128) + (((int)threadIdx.x) & 127))])));
    }
    ((float*)S)[((i_8 * 256) + ((int)threadIdx.x))] = ((((float*)S)[((i_8 * 256) + ((int)threadIdx.x))] * expf(chunk_g)) + acc_3[0]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 16; ++i_9) {
    *(float4*)(FinalState + (((((int)blockIdx.x) * 16384) + (i_9 * 1024)) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)S) + ((i_9 * 1024) + (((int)threadIdx.x) * 4)));
  }
}

