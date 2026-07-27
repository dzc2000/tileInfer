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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ K_cache, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V_cache, const int* __restrict__ block_table, const int* __restrict__ seqlen_kv);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const bfloat16_t* __restrict__ K_cache, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V_cache, const int* __restrict__ block_table, const int* __restrict__ seqlen_kv) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 40960));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 73728));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 75776));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 75776));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 75776));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 76800));
  float acc_o[16];
  float logsum[2];
  float scores_max[2];
  float acc_s[4];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  float scores_max_clear[2];
  bfloat16_t S_shared_local_cast[2];
  float scores_max_clear_1[2];
  bfloat16_t S_shared_local_cast_1[2];
  bfloat16_t O_shared_local_cast_2[2];
  int seqlen_kv_b = seqlen_kv[((int)blockIdx.x)];
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    *(uint1*)(((bfloat16_t*)Q_shared) + (((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (i * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Q + ((((((int)blockIdx.x) * 3072) + (((int)blockIdx.y) * 1536)) + (i * 512)) + (((int)threadIdx.x) * 2)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  __syncthreads();
  if (0 < seqlen_kv_b) {
    for (int sub = 0; sub < 4; ++sub) {
      int physical_page = block_table[((((int)blockIdx.x) * 15922) + sub)];
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)K_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (sub * 1024)) + (i_2 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K_cache[(((((((int64_t)physical_page) * (int64_t)8192) + (((int64_t)i_2) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((physical_page < 15922) && (0 <= physical_page)) && (physical_page < 15922)) && (0 <= physical_page)));
      }
    }
    tl::cp_async_commit();
    for (int sub_1 = 0; sub_1 < 4; ++sub_1) {
      int physical_page_1 = block_table[((((int)blockIdx.x) * 15922) + sub_1)];
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)V_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (sub_1 * 1024)) + (i_3 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V_cache[(((((((int64_t)physical_page_1) * (int64_t)8192) + (((int64_t)i_3) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((physical_page_1 < 15922) && (0 <= physical_page_1)) && (physical_page_1 < 15922)) && (0 <= physical_page_1)));
      }
    }
    tl::cp_async_commit();
  }
  for (int k = 0; k < (((seqlen_kv_b + 63) >> 6) - 1); ++k) {
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[4];
      for (int ki = 0; ki < 16; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_shared)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x2((&(((bfloat16_t*)K_shared)[(((((((ki >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      }
    }
    __syncthreads();
    for (int sub_2 = 0; sub_2 < 4; ++sub_2) {
      int condval;
      if ((((k * 2) + (sub_2 >> 1)) < 7959)) {
        condval = block_table[((((((int)blockIdx.x) * 15922) + (k * 4)) + sub_2) + 4)];
      } else {
        condval = 0;
      }
      int physical_page_2 = condval;
      #pragma unroll
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)K_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (sub_2 * 1024)) + (i_4 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K_cache[(((((((int64_t)physical_page_2) * (int64_t)8192) + (((int64_t)i_4) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((physical_page_2 < 15922) && (0 <= physical_page_2)) && (physical_page_2 < 15922)) && (0 <= physical_page_2)));
      }
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      float condval_1;
      if ((((((i_5 >> 1) * 4) + ((((int)threadIdx.x) & 31) >> 3)) < 3) && (((((k * 64) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_5 & 1)) < seqlen_kv_b))) {
        condval_1 = acc_s[i_5];
      } else {
        condval_1 = -CUDART_INF_F;
      }
      acc_s[i_5] = condval_1;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      scores_max_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 2; ++rv) {
        scores_max_clear[i_6] = max(scores_max_clear[i_6], acc_s[((i_6 * 2) + rv)]);
      }
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 256, 32, 0>::run(scores_max_clear[i_6], (&(((float*)workspace_3)[0])));
      scores_max_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear[i_6]);
      scores_max[i_6] = max(scores_max[i_6], scores_max_clear[i_6]);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_scale[i_7] = exp2f(((scores_max_prev[i_7] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[i_7] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      acc_s[i_8] = exp2f(((acc_s[i_8] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[(i_8 >> 1)] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_sum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
        scores_sum[i_9] = (scores_sum[i_9] + acc_s[((i_9 * 2) + rv_1)]);
      }
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 256, 32, 0>::run(scores_sum[i_9], (&(((float*)workspace_2)[0])));
      scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      logsum[i_10] = ((logsum[i_10] * scores_scale[i_10]) + scores_sum[i_10]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_11 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      *(uint1*)(S_shared_local_cast + 0) = __1;
      *(uint1*)(((bfloat16_t*)S_shared) + ((((((i_11 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast + 0);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      acc_o[i_12] = (acc_o[i_12] * scores_scale[((i_12 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[16];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_shared)[(((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_13 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    for (int sub_3 = 0; sub_3 < 4; ++sub_3) {
      int condval_2;
      if ((((k * 2) + (sub_3 >> 1)) < 7959)) {
        condval_2 = block_table[((((((int)blockIdx.x) * 15922) + (k * 4)) + sub_3) + 4)];
      } else {
        condval_2 = 0;
      }
      int physical_page_3 = condval_2;
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)V_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (sub_3 * 1024)) + (i_14 * 512)) + ((((int)threadIdx.x) >> 5) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V_cache[(((((((int64_t)physical_page_3) * (int64_t)8192) + (((int64_t)i_14) * (int64_t)4096)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((physical_page_3 < 15922) && (0 <= physical_page_3)) && (physical_page_3 < 15922)) && (0 <= physical_page_3)));
      }
    }
    tl::cp_async_commit();
  }
  if (1 <= seqlen_kv_b) {
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  if (1 <= seqlen_kv_b) {
    {
      bfloat16_t A_local_2[8];
      bfloat16_t B_local_2[4];
      for (int ki_2 = 0; ki_2 < 16; ++ki_2) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_shared)[((((ki_2 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
        tl::ptx_ldmatrix_x2((&(((bfloat16_t*)K_shared)[(((((((ki_2 >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + 0), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + 0));
      }
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      float condval_3;
      if ((((((i_15 >> 1) * 4) + ((((int)threadIdx.x) & 31) >> 3)) < 3) && (((((((seqlen_kv_b + 63) >> 6) * 64) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_15 & 1)) < (seqlen_kv_b + 64)))) {
        condval_3 = acc_s[i_15];
      } else {
        condval_3 = -CUDART_INF_F;
      }
      acc_s[i_15] = condval_3;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      scores_max_clear_1[i_16] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
        scores_max_clear_1[i_16] = max(scores_max_clear_1[i_16], acc_s[((i_16 * 2) + rv_2)]);
      }
      scores_max_clear_1[i_16] = tl::AllReduce<tl::MaxOp, 256, 32, 0>::run(scores_max_clear_1[i_16], (&(((float*)workspace_1)[0])));
      scores_max_clear_1[i_16] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear_1[i_16]);
      scores_max[i_16] = max(scores_max[i_16], scores_max_clear_1[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      scores_scale[i_17] = exp2f(((scores_max_prev[i_17] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[i_17] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 4; ++i_18) {
      acc_s[i_18] = exp2f(((acc_s[i_18] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[(i_18 >> 1)] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      scores_sum[i_19] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
        scores_sum[i_19] = (scores_sum[i_19] + acc_s[((i_19 * 2) + rv_3)]);
      }
      scores_sum[i_19] = tl::AllReduce<tl::SumOp, 256, 32, 0>::run(scores_sum[i_19], (&(((float*)workspace)[0])));
      scores_sum[i_19] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      logsum[i_20] = ((logsum[i_20] * scores_scale[i_20]) + scores_sum[i_20]);
    }
  }
  __syncthreads();
  if (1 <= seqlen_kv_b) {
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_s + (i_21 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
      *(uint1*)(S_shared_local_cast_1 + 0) = __2;
      *(uint1*)(((bfloat16_t*)S_shared) + ((((((i_21 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(S_shared_local_cast_1 + 0);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 16; ++i_22) {
      acc_o[i_22] = (acc_o[i_22] * scores_scale[((i_22 & 3) >> 1)]);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  if (1 <= seqlen_kv_b) {
    {
      bfloat16_t A_local_3[8];
      bfloat16_t B_local_3[16];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)S_shared)[((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
        for (int i_23 = 0; i_23 < 2; ++i_23) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_shared)[(((((((int)threadIdx.x) >> 6) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_23) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_3[(i_23 * 8)])));
        }
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 16; ++i_24) {
    float condval_4;
    if (((((((i_24 & 3) >> 1) * 4) + ((((int)threadIdx.x) & 31) >> 3)) < 3) && (0x0p+0f/*0.000000e+00*/ < logsum[((i_24 & 3) >> 1)]))) {
      condval_4 = (acc_o[i_24] / logsum[((i_24 & 3) >> 1)]);
    } else {
      condval_4 = 0x0p+0f/*0.000000e+00*/;
    }
    acc_o[i_24] = condval_4;
  }
  __syncthreads();
  #pragma unroll
  for (int i_25 = 0; i_25 < 8; ++i_25) {
    if ((((i_25 & 1) * 4) + ((((int)threadIdx.x) & 31) >> 3)) < 3) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_o + (i_25 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      *(uint1*)(O_shared_local_cast_2 + 0) = __3;
      *(uint1*)(((bfloat16_t*)O_shared) + ((((((i_25 & 1) * 2048) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_25 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_shared_local_cast_2 + 0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 3; ++i_26) {
    *(uint1*)(Output + ((((((int)blockIdx.x) * 3072) + (((int)blockIdx.y) * 1536)) + (i_26 * 512)) + (((int)threadIdx.x) * 2))) = *(uint1*)(((bfloat16_t*)O_shared) + ((i_26 * 512) + (((int)threadIdx.x) * 2)));
  }
}

