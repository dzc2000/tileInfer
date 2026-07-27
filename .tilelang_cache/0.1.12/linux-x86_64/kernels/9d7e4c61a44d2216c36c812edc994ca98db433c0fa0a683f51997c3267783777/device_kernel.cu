#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, int64_t* __restrict__ partial_idx, float* __restrict__ partial_max);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, int64_t* __restrict__ partial_idx, float* __restrict__ partial_max) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  float C_local[32];
  float row_extreme[8];
  float C_shared_frag[32];
  float row_extreme_clear[8];
  int cand[1024];
  int first_idx[8];
  int64_t out_idx[8];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((i_1 * 163840) + ((((int)threadIdx.x) >> 3) * 5120)) + ((((int)threadIdx.x) & 7) * 8))])), ((((i_1 * 32) + (((int)threadIdx.x) >> 3)) < 1) && (((i_1 * 32) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)B_shared)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((int)blockIdx.x) * 655360) + (i_2 * 163840)) + ((((int)threadIdx.x) >> 3) * 5120)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[((((i_3 * 163840) + ((((int)threadIdx.x) >> 3) * 5120)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])), ((((i_3 * 32) + (((int)threadIdx.x) >> 3)) < 1) && (((i_3 * 32) + (((int)threadIdx.x) >> 3)) < 1)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)B_shared)[((((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[(((((((int)blockIdx.x) * 655360) + (i_4 * 163840)) + ((((int)threadIdx.x) >> 3) * 5120)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 78; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((((k + 2) % 3) * 4096) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((i_5 * 163840) + ((((int)threadIdx.x) >> 3) * 5120)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])), ((((i_5 * 32) + (((int)threadIdx.x) >> 3)) < 1) && (((i_5 * 32) + (((int)threadIdx.x) >> 3)) < 1)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)B_shared)[((((((((k + 2) % 3) * 8192) + (i_6 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int)blockIdx.x) * 655360) + (i_6 * 163840)) + ((((int)threadIdx.x) >> 3) * 5120)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      bfloat16_t A_local[16];
      bfloat16_t B_local[16];
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)A_shared)[((((((k % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_7 * 8)])));
        }
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)B_shared)[(((((((((k % 3) * 8192) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_8 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_8 * 8)])));
        }
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_9 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_9 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    bfloat16_t A_local_1[16];
    bfloat16_t B_local_1[16];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)A_shared)[((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[(i_10 * 8)])));
      }
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (i_11 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_11 * 8)])));
      }
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_12 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_12 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    bfloat16_t A_local_2[16];
    bfloat16_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)A_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])), (&(A_local_2[(i_13 * 8)])));
      }
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)B_shared)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_14 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_2[(i_14 * 8)])));
      }
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_15 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_15 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    *(float2*)(((float*)C_shared) + (((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_16 >> 3) * 2048)) + ((i_16 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_16 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_16 * 2));
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    float broadcast_var_1 = -CUDART_INF_F;
    *(float4*)(row_extreme + (i_17 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 8; ++i_18) {
    *(float4*)(C_shared_frag + (i_18 * 4)) = *(float4*)(((float*)C_shared) + ((i_18 * 1024) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    row_extreme_clear[i_19] = -CUDART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      row_extreme_clear[i_19] = max(row_extreme_clear[i_19], C_shared_frag[((i_19 * 4) + rv)]);
    }
    row_extreme_clear[i_19] = tl::AllReduce<tl::MaxOp, 32, 1, 0>::run(row_extreme_clear[i_19]);
    row_extreme[i_19] = max(row_extreme[i_19], row_extreme_clear[i_19]);
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 1024; ++i_20) {
    if (((float*)C_shared)[((((i_20 >> 7) * 1024) + ((((int)threadIdx.x) >> 5) * 128)) + (i_20 & 127))] == row_extreme[(i_20 >> 7)]) {
      cand[i_20] = (i_20 & 127);
    } else {
      cand[i_20] = 128;
    }
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 8; ++i_21) {
    first_idx[i_21] = 2147483647;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 128; ++rv_1) {
      first_idx[i_21] = min(first_idx[i_21], cand[((i_21 * 128) + rv_1)]);
    }
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 8; ++i_22) {
    out_idx[i_22] = ((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)first_idx[i_22]));
  }
  if ((((int)threadIdx.x) % 32) == 0) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      if (((i_23 * 8) + (((int)threadIdx.x) >> 5)) < 1) {
        partial_max[(((i_23 * 8) + (((int)threadIdx.x) >> 5)) + ((int)blockIdx.x))] = row_extreme[i_23];
      }
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      if (((i_24 * 8) + (((int)threadIdx.x) >> 5)) < 1) {
        partial_idx[(((i_24 * 8) + (((int)threadIdx.x) >> 5)) + ((int)blockIdx.x))] = out_idx[i_24];
      }
    }
  }
}

