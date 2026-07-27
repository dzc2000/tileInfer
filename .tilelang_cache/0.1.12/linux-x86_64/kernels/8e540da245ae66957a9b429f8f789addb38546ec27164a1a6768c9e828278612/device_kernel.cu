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

extern "C" __global__ void main_kernel(const bfloat16_t* __restrict__ K_unpad, bfloat16_t* __restrict__ Output_unpad, const bfloat16_t* __restrict__ Q_unpad, const bfloat16_t* __restrict__ V_unpad, const int* __restrict__ cu_seqlens_k, const int* __restrict__ cu_seqlens_q, int max_seqlen_q);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const bfloat16_t* __restrict__ K_unpad, bfloat16_t* __restrict__ Output_unpad, const bfloat16_t* __restrict__ Q_unpad, const bfloat16_t* __restrict__ V_unpad, const int* __restrict__ cu_seqlens_k, const int* __restrict__ cu_seqlens_q, int max_seqlen_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 98304));
  float acc_o[128];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_s_bf16[32];
  bfloat16_t Output_unpad_local_cast[2];
  int q_start_idx = cu_seqlens_q[0];
  int kv_start_idx = cu_seqlens_k[0];
  int q_end_idx = cu_seqlens_q[1];
  int k_end_idx = cu_seqlens_k[1];
  bool is_valid_cta = ((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx));
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((((int)blockIdx.x) * 64) + (i * 4)) + (((int)threadIdx.x) >> 5)) + q_start_idx) < 205) && (0 <= ((((((int)blockIdx.x) * 64) + (i * 4)) + (((int)threadIdx.x) >> 5)) + q_start_idx)))) {
      condval = *(uint4*)(Q_unpad + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)196608) + (((int64_t)i) * (int64_t)12288)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)3072)) + (((int64_t)q_start_idx) * (int64_t)3072)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)Q_shared) + ((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  __syncthreads();
  int condval_1;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_1 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_1 = 0;
  }
  if (0 < condval_1) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)K_shared)[((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_2 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K_unpad[(((((((int64_t)i_2) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)6) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), (((((((i_2 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 205) && (0 <= (((i_2 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))) && ((((i_2 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 205)) && (0 <= (((i_2 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)V_shared)[((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_3 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V_unpad[(((((((int64_t)i_3) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)6) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), (((((((i_3 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 205) && (0 <= (((i_3 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))) && ((((i_3 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 205)) && (0 <= (((i_3 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))));
    }
    tl::cp_async_commit();
  }
  int condval_2;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_2 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_2 = 0;
  }
  if (1 < condval_2) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)K_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_4 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_4 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(K_unpad[((((((((int64_t)i_4) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)6) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (int64_t)32768)])), (((((((i_4 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 141) && (-64 <= (((i_4 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))) && ((((i_4 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 141)) && (-64 <= (((i_4 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)V_shared)[(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_5 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(V_unpad[((((((((int64_t)i_5) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)6) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (int64_t)32768)])), (((((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 141) && (-64 <= (((i_5 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))) && ((((i_5 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 141)) && (-64 <= (((i_5 * 4) + (((int)threadIdx.x) >> 5)) + kv_start_idx))));
    }
    tl::cp_async_commit();
  }
  int condval_3;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_3 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_3 = 0;
  }
  for (int k = 0; k < (condval_3 - 2); ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[32];
      for (int ki = 0; ki < 16; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_shared)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)K_shared)[(((((((((k & 1) * 16384) + ((ki >> 2) * 4096)) + (i_7 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_7 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)K_shared)[((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 31) >> 3) * 4096)) + (i_8 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K_unpad[(((((((((int64_t)k) * (int64_t)32768) + (((int64_t)i_8) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)6) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (int64_t)65536)])), ((((((((k * 64) + (i_8 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 77) && (-128 <= ((((k * 64) + (i_8 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx))) && (((((k * 64) + (i_8 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 77)) && (-128 <= ((((k * 64) + (i_8 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx))));
    }
    tl::cp_async_commit();
    bool need_mask = ((((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) - kv_start_idx) - q_end_idx) < ((k * 64) + 63)) | ((q_end_idx - q_start_idx) < ((((int)blockIdx.x) * 64) + 64))) | ((k_end_idx - kv_start_idx) < ((k * 64) + 64)));
    if ((((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) - kv_start_idx) - q_end_idx) < ((k * 64) + 63)) | ((q_end_idx - q_start_idx) < ((((int)blockIdx.x) * 64) + 64))) | ((k_end_idx - kv_start_idx) < ((k * 64) + 64))) {
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        float condval_4;
        if ((((((((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_9 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + k_end_idx) + q_start_idx) - kv_start_idx) - q_end_idx) < ((((k * 64) + ((i_9 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1))) | ((q_end_idx - q_start_idx) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_9 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) | ((k_end_idx - kv_start_idx) <= ((((k * 64) + ((i_9 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1))))) {
          condval_4 = -CUDART_INF_F;
        } else {
          condval_4 = acc_s[i_9];
        }
        acc_s[i_9] = condval_4;
      }
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_max[i_10] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max[i_10] = max(scores_max[i_10], acc_s[((((rv & 7) * 4) + (i_10 * 2)) + (rv >> 3))]);
      }
      scores_max[i_10] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_max[i_11] = max(scores_max[i_11], scores_max_prev[i_11]);
      scores_scale[i_11] = exp2f(((scores_max_prev[i_11] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[i_11] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      acc_s[i_12] = exp2f(((acc_s[i_12] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[((i_12 & 3) >> 1)] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_sum[i_13] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_13] = (scores_sum[i_13] + acc_s[((((rv_1 & 7) * 4) + (i_13 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_13] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      logsum[i_14] = ((logsum[i_14] * scores_scale[i_14]) + scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 128; ++i_15) {
      acc_o[i_15] = (acc_o[i_15] * scores_scale[((i_15 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_16 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_bf16 + (i_16 * 4)) = __1;
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      bfloat16_t B_local_1[128];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_17 = 0; i_17 < 16; ++i_17) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_shared)[((((((k & 1) * 16384) + ((i_17 >> 2) * 4096)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_1[(i_17 * 8)])));
        }
        for (int j_1 = 0; j_1 < 16; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_bf16 + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_bf16 + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)V_shared)[((((((((k & 1) * 16384) + (((((int)threadIdx.x) & 31) >> 3) * 4096)) + (i_18 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_18 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V_unpad[(((((((((int64_t)k) * (int64_t)32768) + (((int64_t)i_18) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)512)) + (((int64_t)kv_start_idx) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)6) * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (int64_t)65536)])), ((((((((k * 64) + (i_18 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 77) && (-128 <= ((((k * 64) + (i_18 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx))) && (((((k * 64) + (i_18 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx) < 77)) && (-128 <= ((((k * 64) + (i_18 * 4)) + (((int)threadIdx.x) >> 5)) + kv_start_idx))));
    }
    tl::cp_async_commit();
  }
  int condval_5;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_5 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_5 = 0;
  }
  if (2 <= condval_5) {
    #pragma unroll
    for (int i_19 = 0; i_19 < 8; ++i_19) {
      float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_19 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  int condval_6;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_6 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_6 = 0;
  }
  if (2 <= condval_6) {
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_2[32];
      for (int ki_2 = 0; ki_2 < 16; ++ki_2) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_shared)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_1[0])));
        for (int i_20 = 0; i_20 < 4; ++i_20) {
          int condval_7;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_7 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_7 = 0;
          }
          int condval_8;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_8 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_8 = 0;
          }
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)K_shared)[(((((((((condval_8 & 1) * 16384) + ((ki_2 >> 2) * 4096)) + (i_20 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_20 * 8)])));
        }
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
    int condval_9;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_9 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_9 = 0;
    }
    int condval_10;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_10 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_10 = 0;
    }
    bool need_mask_1 = (((((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 65) - kv_start_idx) - q_end_idx) < (condval_9 * 64)) | ((q_end_idx - q_start_idx) < ((((int)blockIdx.x) * 64) + 64))) | (((k_end_idx + 64) - kv_start_idx) < (condval_10 * 64)));
    int condval_11;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_11 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_11 = 0;
    }
    int condval_12;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_12 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_12 = 0;
    }
    if (((((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 65) - kv_start_idx) - q_end_idx) < (condval_11 * 64)) | ((q_end_idx - q_start_idx) < ((((int)blockIdx.x) * 64) + 64))) | (((k_end_idx + 64) - kv_start_idx) < (condval_12 * 64))) {
      #pragma unroll
      for (int i_21 = 0; i_21 < 32; ++i_21) {
        int condval_14;
        if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
          condval_14 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
        } else {
          condval_14 = 0;
        }
        int condval_15;
        if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
          condval_15 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
        } else {
          condval_15 = 0;
        }
        float condval_13;
        if (((((((((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_21 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + k_end_idx) + q_start_idx) + 128) - kv_start_idx) - q_end_idx) < ((((condval_14 * 64) + ((i_21 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_21 & 1))) | ((q_end_idx - q_start_idx) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_21 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) | (((k_end_idx + 128) - kv_start_idx) <= ((((condval_15 * 64) + ((i_21 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_21 & 1))))) {
          condval_13 = -CUDART_INF_F;
        } else {
          condval_13 = acc_s[i_21];
        }
        acc_s[i_21] = condval_13;
      }
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      scores_max[i_22] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
        scores_max[i_22] = max(scores_max[i_22], acc_s[((((rv_2 & 7) * 4) + (i_22 * 2)) + (rv_2 >> 3))]);
      }
      scores_max[i_22] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_22]);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_max[i_23] = max(scores_max[i_23], scores_max_prev[i_23]);
      scores_scale[i_23] = exp2f(((scores_max_prev[i_23] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[i_23] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 32; ++i_24) {
      acc_s[i_24] = exp2f(((acc_s[i_24] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[((i_24 & 3) >> 1)] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_sum[i_25] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
        scores_sum[i_25] = (scores_sum[i_25] + acc_s[((((rv_3 & 7) * 4) + (i_25 * 2)) + (rv_3 >> 3))]);
      }
      scores_sum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_25]);
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      logsum[i_26] = ((logsum[i_26] * scores_scale[i_26]) + scores_sum[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 128; ++i_27) {
      acc_o[i_27] = (acc_o[i_27] * scores_scale[((i_27 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 8; ++i_28) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_s + (i_28 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
      *(uint2*)(acc_s_bf16 + (i_28 * 4)) = __2;
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  int condval_16;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_16 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_16 = 0;
  }
  if (2 <= condval_16) {
    {
      bfloat16_t B_local_3[128];
      for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
        for (int i_29 = 0; i_29 < 16; ++i_29) {
          int condval_17;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_17 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_17 = 0;
          }
          int condval_18;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_18 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_18 = 0;
          }
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_shared)[((((((condval_18 & 1) * 16384) + ((i_29 >> 2) * 4096)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_3[(i_29 * 8)])));
        }
        for (int j_3 = 0; j_3 < 16; ++j_3) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_bf16 + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_bf16 + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  int condval_19;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_19 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_19 = 0;
  }
  if (1 <= condval_19) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 8; ++i_30) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_30 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  int condval_20;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_20 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_20 = 0;
  }
  if (1 <= condval_20) {
    {
      bfloat16_t A_local_2[8];
      bfloat16_t B_local_4[32];
      for (int ki_4 = 0; ki_4 < 16; ++ki_4) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)Q_shared)[(((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
        for (int i_31 = 0; i_31 < 4; ++i_31) {
          int condval_21;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_21 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_21 = 0;
          }
          int condval_22;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_22 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_22 = 0;
          }
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)K_shared)[((((((((((condval_22 + 1) & 1) * 16384) + ((ki_4 >> 2) * 4096)) + (i_31 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_4[(i_31 * 8)])));
        }
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
        }
      }
    }
    int condval_23;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_23 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_23 = 0;
    }
    int condval_24;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_24 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_24 = 0;
    }
    bool need_mask_2 = (((((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 1) - kv_start_idx) - q_end_idx) < (condval_23 * 64)) | ((q_end_idx - q_start_idx) < ((((int)blockIdx.x) * 64) + 64))) | ((k_end_idx - kv_start_idx) < (condval_24 * 64)));
    int condval_25;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_25 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_25 = 0;
    }
    int condval_26;
    if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
      condval_26 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
    } else {
      condval_26 = 0;
    }
    if (((((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 1) - kv_start_idx) - q_end_idx) < (condval_25 * 64)) | ((q_end_idx - q_start_idx) < ((((int)blockIdx.x) * 64) + 64))) | ((k_end_idx - kv_start_idx) < (condval_26 * 64))) {
      #pragma unroll
      for (int i_32 = 0; i_32 < 32; ++i_32) {
        int condval_28;
        if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
          condval_28 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
        } else {
          condval_28 = 0;
        }
        int condval_29;
        if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
          condval_29 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
        } else {
          condval_29 = 0;
        }
        float condval_27;
        if (((((((((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_32 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + k_end_idx) + q_start_idx) + 64) - kv_start_idx) - q_end_idx) < ((((condval_28 * 64) + ((i_32 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_32 & 1))) | ((q_end_idx - q_start_idx) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_32 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) | (((k_end_idx + 64) - kv_start_idx) <= ((((condval_29 * 64) + ((i_32 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_32 & 1))))) {
          condval_27 = -CUDART_INF_F;
        } else {
          condval_27 = acc_s[i_32];
        }
        acc_s[i_32] = condval_27;
      }
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      scores_max[i_33] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        scores_max[i_33] = max(scores_max[i_33], acc_s[((((rv_4 & 7) * 4) + (i_33 * 2)) + (rv_4 >> 3))]);
      }
      scores_max[i_33] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_33]);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 2; ++i_34) {
      scores_max[i_34] = max(scores_max[i_34], scores_max_prev[i_34]);
      scores_scale[i_34] = exp2f(((scores_max_prev[i_34] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[i_34] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 32; ++i_35) {
      acc_s[i_35] = exp2f(((acc_s[i_35] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/) - (scores_max[((i_35 & 3) >> 1)] * 0x1.7154764ee6c2fp-4f/*9.016844e-02*/)));
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      scores_sum[i_36] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
        scores_sum[i_36] = (scores_sum[i_36] + acc_s[((((rv_5 & 7) * 4) + (i_36 * 2)) + (rv_5 >> 3))]);
      }
      scores_sum[i_36] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_36]);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      logsum[i_37] = ((logsum[i_37] * scores_scale[i_37]) + scores_sum[i_37]);
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 128; ++i_38) {
      acc_o[i_38] = (acc_o[i_38] * scores_scale[((i_38 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 8; ++i_39) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc_s + (i_39 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(acc_s_bf16 + (i_39 * 4)) = __3;
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  int condval_30;
  if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
    condval_30 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
  } else {
    condval_30 = 0;
  }
  if (1 <= condval_30) {
    {
      bfloat16_t B_local_5[128];
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        for (int i_40 = 0; i_40 < 16; ++i_40) {
          int condval_31;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_31 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_31 = 0;
          }
          int condval_32;
          if (((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx))) {
            condval_32 = (min(((((((((int)blockIdx.x) * 64) + k_end_idx) + q_start_idx) + 127) - kv_start_idx) - q_end_idx), ((k_end_idx + 63) - kv_start_idx)) >> 6);
          } else {
            condval_32 = 0;
          }
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)V_shared)[(((((((condval_32 + 1) & 1) * 16384) + ((i_40 >> 2) * 4096)) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_40 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_40 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_5[(i_40 * 8)])));
        }
        for (int j_5 = 0; j_5 < 16; ++j_5) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_bf16 + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_bf16 + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
        }
      }
    }
  }
  if ((((int)blockIdx.x) * 64) < (q_end_idx - q_start_idx)) {
    #pragma unroll
    for (int i_41 = 0; i_41 < 64; ++i_41) {
      if (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_41 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < (q_end_idx - q_start_idx)) {
        float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
        uint1 __4;
        float2 condval_33;
        if ((logsum[(i_41 & 1)] == 0x0p+0f/*0.000000e+00*/)) {
          condval_33 = make_float2(broadcast_var_7, broadcast_var_7);
        } else {
          float2 __5;
            float2 v__3 = *(float2*)(acc_o + (i_41 * 2));
            float2 v__4 = make_float2(logsum[(i_41 & 1)], logsum[(i_41 & 1)]);
            __5.x = (v__3.x/v__4.x);
            __5.y = (v__3.y/v__4.y);
          condval_33 = __5;
        }
        (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&condval_33))[0]);
        *(uint1*)(Output_unpad_local_cast + 0) = __4;
        if (0 <= (((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_41 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + q_start_idx)) {
          if ((((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_41 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + q_start_idx) < 205) {
            *(uint1*)(Output_unpad + ((((((((((int64_t)((int)blockIdx.x)) * (int64_t)196608) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)49152)) + ((((int64_t)i_41) & (int64_t)1) * (int64_t)24576)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)3072)) + (((int64_t)q_start_idx) * (int64_t)3072)) + (((int64_t)((int)blockIdx.y)) * (int64_t)256)) + ((((int64_t)i_41) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(Output_unpad_local_cast + 0);
          }
        }
      }
    }
  }
}

