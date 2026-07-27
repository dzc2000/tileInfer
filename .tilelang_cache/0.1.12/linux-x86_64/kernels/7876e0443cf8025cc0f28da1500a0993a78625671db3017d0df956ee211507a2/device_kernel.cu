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

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ Alpha, const bfloat16_t* __restrict__ BetaRaw, const bfloat16_t* __restrict__ DtBias, const bfloat16_t* __restrict__ NegAExp, bfloat16_t* __restrict__ NewState, const bfloat16_t* __restrict__ NormW, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ QKV, const bfloat16_t* __restrict__ State, const bfloat16_t* __restrict__ Z);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const bfloat16_t* __restrict__ Alpha, const bfloat16_t* __restrict__ BetaRaw, const bfloat16_t* __restrict__ DtBias, const bfloat16_t* __restrict__ NegAExp, bfloat16_t* __restrict__ NewState, const bfloat16_t* __restrict__ NormW, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ QKV, const bfloat16_t* __restrict__ State, const bfloat16_t* __restrict__ Z) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* z_val = ((void*)((char*)buf_dyn_shmem + 0));
  void* var_buf = ((void*)((char*)buf_dyn_shmem + 512));
  void* final_output = ((void*)((char*)buf_dyn_shmem + 528));
  void* norm_w = ((void*)((char*)buf_dyn_shmem + 528));
  void* out_sq = ((void*)((char*)buf_dyn_shmem + 528));
  void* q_full = ((void*)((char*)buf_dyn_shmem + 528));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 528));
  void* k_full = ((void*)((char*)buf_dyn_shmem + 784));
  void* output = ((void*)((char*)buf_dyn_shmem + 1040));
  void* v = ((void*)((char*)buf_dyn_shmem + 1040));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 1040));
  void* accum_sk = ((void*)((char*)buf_dyn_shmem + 1296));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 1296));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 1296));
  void* o_normed = ((void*)((char*)buf_dyn_shmem + 1552));
  void* accum_sq = ((void*)((char*)buf_dyn_shmem + 1808));
  void* S_tile = ((void*)((char*)buf_dyn_shmem + 2320));
  void* delta = ((void*)((char*)buf_dyn_shmem + 2320));
  void* S_tile2 = ((void*)((char*)buf_dyn_shmem + 2832));
  void* S_decayed_tile = ((void*)((char*)buf_dyn_shmem + 10512));
  void* k_tile = ((void*)((char*)buf_dyn_shmem + 18704));
  void* q_tile = ((void*)((char*)buf_dyn_shmem + 18736));
  void* qk_dot = ((void*)((char*)buf_dyn_shmem + 18736));
  float pow_frag[1];
  float pow_sum_frag[1];
  float dot_k[1];
  float dot_q[1];
  bfloat16_t S_tile_local_cast_1[4];
  float S_decayed_tile_local_cast[4];
  bfloat16_t S_tile_local_cast_3[4];
  float S_decayed_tile_local_cast_2[4];
  bfloat16_t S_tile_local_cast_5[4];
  float S_decayed_tile_local_cast_4[4];
  float qk_frag[1];
  float qk_sum[1];
  bfloat16_t S_tile2_local_cast_7[4];
  bfloat16_t k_tile_local_cast_8[4];
  float delta_local_cast_9[4];
  bfloat16_t NewState_local_cast_6[4];
  bfloat16_t S_tile2_local_cast_11[4];
  bfloat16_t k_tile_local_cast_12[4];
  float delta_local_cast_13[4];
  bfloat16_t NewState_local_cast_10[4];
  float out_sq_frag[1];
  float var_buf_frag[1];
  ((bfloat16_t*)q_full)[((int)threadIdx.x)] = QKV[(((((int)blockIdx.y) * 5120) + ((((int)blockIdx.x) / 3) * 128)) + ((int)threadIdx.x))];
  ((bfloat16_t*)k_full)[((int)threadIdx.x)] = QKV[((((((int)blockIdx.y) * 5120) + ((((int)blockIdx.x) / 3) * 128)) + ((int)threadIdx.x)) + 1024)];
  ((bfloat16_t*)v)[((int)threadIdx.x)] = QKV[((((((int)blockIdx.y) * 5120) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)) + 2048)];
  pow_frag[0] = (((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]) * ((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]));
  pow_sum_frag[0] = 0x0p+0f/*0.000000e+00*/;
  pow_sum_frag[0] = (pow_sum_frag[0] + pow_frag[0]);
  __syncthreads();
  pow_sum_frag[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(pow_sum_frag[0], (&(((float*)workspace_2)[0])));
  float q_norm = max(sqrtf(pow_sum_frag[0]), 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  __syncthreads();
  ((bfloat16_t*)q_full)[((int)threadIdx.x)] = ((bfloat16_t)((((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]) / q_norm) * 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/));
  pow_frag[0] = (((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]) * ((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]));
  pow_sum_frag[0] = 0x0p+0f/*0.000000e+00*/;
  pow_sum_frag[0] = (pow_sum_frag[0] + pow_frag[0]);
  __syncthreads();
  pow_sum_frag[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(pow_sum_frag[0], (&(((float*)workspace_3)[0])));
  float k_norm = max(sqrtf(pow_sum_frag[0]), 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  __syncthreads();
  ((bfloat16_t*)k_full)[((int)threadIdx.x)] = ((bfloat16_t)(((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]) / k_norm));
  float alpha_val = ((float)Alpha[((((int)blockIdx.y) * 24) + ((int)blockIdx.x))]);
  float neg_a_exp_val = ((float)NegAExp[((int)blockIdx.x)]);
  float dt_bias_val = ((float)DtBias[((int)blockIdx.x)]);
  float sp_input = (alpha_val + dt_bias_val);
  float condval;
  if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
    condval = (alpha_val + dt_bias_val);
  } else {
    condval = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
  }
  float sp = condval;
  float condval_1;
  if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
    condval_1 = (alpha_val + dt_bias_val);
  } else {
    condval_1 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
  }
  float gate = (neg_a_exp_val * condval_1);
  float condval_2;
  if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
    condval_2 = (alpha_val + dt_bias_val);
  } else {
    condval_2 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
  }
  float decay = expf((neg_a_exp_val * condval_2));
  float beta = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((float)BetaRaw[((((int)blockIdx.y) * 24) + ((int)blockIdx.x))])))));
  ((float*)accum_sk)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  ((float*)accum_sq)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_tile)[((i * 1024) + (((int)threadIdx.x) * 8))])), (&(State[((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (i * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_tile)[(((i_1 * 1024) + (((int)threadIdx.x) * 8)) + 2048)])), (&(State[(((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (i_1 * 1024)) + (((int)threadIdx.x) * 8)) + 2048)])));
  }
  tl::cp_async_commit();
  for (int r_start = 0; r_start < 6; ++r_start) {
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[((r_start * 16) + ((int)threadIdx.x))];
      ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[((r_start * 16) + ((int)threadIdx.x))];
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(uint2*)(S_tile_local_cast_1 + 0) = *(uint2*)(((bfloat16_t*)S_tile) + ((((r_start & 1) * 2048) + (i_2 * 512)) + (((int)threadIdx.x) * 4)));
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(S_tile_local_cast_1 + 0);
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
        float condval_3;
        if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
          condval_3 = (alpha_val + dt_bias_val);
        } else {
          condval_3 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
        }
        float4 v__1 = make_float4(expf((neg_a_exp_val * condval_3)), expf((neg_a_exp_val * condval_3)), expf((neg_a_exp_val * condval_3)), expf((neg_a_exp_val * condval_3)));
        __1.x = (__2.x*v__1.x);
        __1.y = (__2.y*v__1.y);
        __1.z = (__2.z*v__1.z);
        __1.w = (__2.w*v__1.w);
      *(float4*)(S_decayed_tile_local_cast + 0) = __1;
      *(float4*)(((float*)S_decayed_tile) + ((i_2 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_tile_local_cast + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)S_tile)[((((r_start & 1) * 2048) + (i_3 * 1024)) + (((int)threadIdx.x) * 8))])), (&(State[((((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (r_start * 2048)) + (i_3 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
    }
    tl::cp_async_commit();
    dot_k[0] = 0x0p+0f/*0.000000e+00*/;
    dot_q[0] = 0x0p+0f/*0.000000e+00*/;
    __syncthreads();
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      float s_val = ((float*)S_decayed_tile)[((i_4 * 128) + ((int)threadIdx.x))];
      dot_k[0] = (dot_k[0] + (s_val * ((float)((bfloat16_t*)k_tile)[i_4])));
      dot_q[0] = (dot_q[0] + (s_val * ((float)((bfloat16_t*)q_tile)[i_4])));
    }
    ((float*)accum_sk)[((int)threadIdx.x)] = (((float*)accum_sk)[((int)threadIdx.x)] + dot_k[0]);
    ((float*)accum_sq)[((int)threadIdx.x)] = (((float*)accum_sq)[((int)threadIdx.x)] + dot_q[0]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 96)];
    ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[(((int)threadIdx.x) + 96)];
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    *(uint2*)(S_tile_local_cast_3 + 0) = *(uint2*)(((bfloat16_t*)S_tile) + ((i_5 * 512) + (((int)threadIdx.x) * 4)));
    float4 __3;
      float4 __4;
      uint2 v__2 = *(uint2*)(S_tile_local_cast_3 + 0);
      ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[0]);
      ((float2*)(&__4))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__2))[1]);
      float condval_4;
      if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
        condval_4 = (alpha_val + dt_bias_val);
      } else {
        condval_4 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
      }
      float4 v__3 = make_float4(expf((neg_a_exp_val * condval_4)), expf((neg_a_exp_val * condval_4)), expf((neg_a_exp_val * condval_4)), expf((neg_a_exp_val * condval_4)));
      __3.x = (__4.x*v__3.x);
      __3.y = (__4.y*v__3.y);
      __3.z = (__4.z*v__3.z);
      __3.w = (__4.w*v__3.w);
    *(float4*)(S_decayed_tile_local_cast_2 + 0) = __3;
    *(float4*)(((float*)S_decayed_tile) + ((i_5 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_tile_local_cast_2 + 0);
  }
  dot_k[0] = 0x0p+0f/*0.000000e+00*/;
  dot_q[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    float s_val_1 = ((float*)S_decayed_tile)[((i_6 * 128) + ((int)threadIdx.x))];
    dot_k[0] = (dot_k[0] + (s_val_1 * ((float)((bfloat16_t*)k_tile)[i_6])));
    dot_q[0] = (dot_q[0] + (s_val_1 * ((float)((bfloat16_t*)q_tile)[i_6])));
  }
  ((float*)accum_sk)[((int)threadIdx.x)] = (((float*)accum_sk)[((int)threadIdx.x)] + dot_k[0]);
  ((float*)accum_sq)[((int)threadIdx.x)] = (((float*)accum_sq)[((int)threadIdx.x)] + dot_q[0]);
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 112)];
    ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[(((int)threadIdx.x) + 112)];
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    *(uint2*)(S_tile_local_cast_5 + 0) = *(uint2*)(((bfloat16_t*)S_tile) + (((i_7 * 512) + (((int)threadIdx.x) * 4)) + 2048));
    float4 __5;
      float4 __6;
      uint2 v__4 = *(uint2*)(S_tile_local_cast_5 + 0);
      ((float2*)(&__6))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
      ((float2*)(&__6))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
      float condval_5;
      if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
        condval_5 = (alpha_val + dt_bias_val);
      } else {
        condval_5 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
      }
      float4 v__5 = make_float4(expf((neg_a_exp_val * condval_5)), expf((neg_a_exp_val * condval_5)), expf((neg_a_exp_val * condval_5)), expf((neg_a_exp_val * condval_5)));
      __5.x = (__6.x*v__5.x);
      __5.y = (__6.y*v__5.y);
      __5.z = (__6.z*v__5.z);
      __5.w = (__6.w*v__5.w);
    *(float4*)(S_decayed_tile_local_cast_4 + 0) = __5;
    *(float4*)(((float*)S_decayed_tile) + ((i_7 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_tile_local_cast_4 + 0);
  }
  dot_k[0] = 0x0p+0f/*0.000000e+00*/;
  dot_q[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    float s_val_2 = ((float*)S_decayed_tile)[((i_8 * 128) + ((int)threadIdx.x))];
    dot_k[0] = (dot_k[0] + (s_val_2 * ((float)((bfloat16_t*)k_tile)[i_8])));
    dot_q[0] = (dot_q[0] + (s_val_2 * ((float)((bfloat16_t*)q_tile)[i_8])));
  }
  ((float*)accum_sk)[((int)threadIdx.x)] = (((float*)accum_sk)[((int)threadIdx.x)] + dot_k[0]);
  ((float*)accum_sq)[((int)threadIdx.x)] = (((float*)accum_sq)[((int)threadIdx.x)] + dot_q[0]);
  ((float*)delta)[((int)threadIdx.x)] = (beta * (((float)((bfloat16_t*)v)[((int)threadIdx.x)]) - ((float*)accum_sk)[((int)threadIdx.x)]));
  qk_frag[0] = (((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]) * ((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]));
  qk_sum[0] = 0x0p+0f/*0.000000e+00*/;
  qk_sum[0] = (qk_sum[0] + qk_frag[0]);
  __syncthreads();
  qk_sum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(qk_sum[0], (&(((float*)workspace_1)[0])));
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float*)qk_dot)[0] = qk_sum[0];
  }
  __syncthreads();
  ((float*)output)[((int)threadIdx.x)] = (((float*)accum_sq)[((int)threadIdx.x)] + (((float*)qk_dot)[0] * ((float*)delta)[((int)threadIdx.x)]));
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 2; ++i_9) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_tile2)[((i_9 * 1024) + (((int)threadIdx.x) * 8))])), (&(State[((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (i_9 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  for (int r_start_1 = 0; r_start_1 < 7; ++r_start_1) {
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)S_tile2)[(((((r_start_1 + 1) & 1) * 2048) + (i_10 * 1024)) + (((int)threadIdx.x) * 8))])), (&(State[((((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (r_start_1 * 2048)) + (i_10 * 1024)) + (((int)threadIdx.x) * 8)) + 2048)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[((r_start_1 * 16) + ((int)threadIdx.x))];
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      *(uint2*)(S_tile2_local_cast_7 + 0) = *(uint2*)(((bfloat16_t*)S_tile2) + ((((r_start_1 & 1) * 2048) + (i_11 * 512)) + (((int)threadIdx.x) * 4)));
      *(uint2*)(k_tile_local_cast_8 + 0) = make_uint2(__pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_11 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_11 * 4) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_11 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_11 * 4) + (((int)threadIdx.x) >> 5))]));
      *(float4*)(delta_local_cast_9 + 0) = *(float4*)(((float*)delta) + ((((int)threadIdx.x) & 31) * 4));
      uint2 __7;
      float4 __8;
        float4 __9;
          float4 __10;
          uint2 v__6 = *(uint2*)(S_tile2_local_cast_7 + 0);
          ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[0]);
          ((float2*)(&__10))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[1]);
          float condval_6;
          if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
            condval_6 = (alpha_val + dt_bias_val);
          } else {
            condval_6 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
          }
          float4 v__7 = make_float4(expf((neg_a_exp_val * condval_6)), expf((neg_a_exp_val * condval_6)), expf((neg_a_exp_val * condval_6)), expf((neg_a_exp_val * condval_6)));
          __9.x = (__10.x*v__7.x);
          __9.y = (__10.y*v__7.y);
          __9.z = (__10.z*v__7.z);
          __9.w = (__10.w*v__7.w);
        float4 __11;
          float4 __12;
          uint2 v__8 = *(uint2*)(k_tile_local_cast_8 + 0);
          ((float2*)(&__12))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[0]);
          ((float2*)(&__12))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[1]);
          float4 v__9 = *(float4*)(delta_local_cast_9 + 0);
          __11.x = (__12.x*v__9.x);
          __11.y = (__12.y*v__9.y);
          __11.z = (__12.z*v__9.z);
          __11.w = (__12.w*v__9.w);
        __8.x = (__9.x+__11.x);
        __8.y = (__9.y+__11.y);
        __8.z = (__9.z+__11.z);
        __8.w = (__9.w+__11.w);
      (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&__8))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&__8))[1]);
      *(uint2*)(NewState_local_cast_6 + 0) = __7;
      *(uint2*)(NewState + (((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (r_start_1 * 2048)) + (i_11 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(NewState_local_cast_6 + 0);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 112)];
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 4; ++i_12) {
    *(uint2*)(S_tile2_local_cast_11 + 0) = *(uint2*)(((bfloat16_t*)S_tile2) + (((i_12 * 512) + (((int)threadIdx.x) * 4)) + 2048));
    *(uint2*)(k_tile_local_cast_12 + 0) = make_uint2(__pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_12 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_12 * 4) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_12 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_12 * 4) + (((int)threadIdx.x) >> 5))]));
    *(float4*)(delta_local_cast_13 + 0) = *(float4*)(((float*)delta) + ((((int)threadIdx.x) & 31) * 4));
    uint2 __13;
    float4 __14;
      float4 __15;
        float4 __16;
        uint2 v__10 = *(uint2*)(S_tile2_local_cast_11 + 0);
        ((float2*)(&__16))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__10))[0]);
        ((float2*)(&__16))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__10))[1]);
        float condval_7;
        if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
          condval_7 = (alpha_val + dt_bias_val);
        } else {
          condval_7 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
        }
        float4 v__11 = make_float4(expf((neg_a_exp_val * condval_7)), expf((neg_a_exp_val * condval_7)), expf((neg_a_exp_val * condval_7)), expf((neg_a_exp_val * condval_7)));
        __15.x = (__16.x*v__11.x);
        __15.y = (__16.y*v__11.y);
        __15.z = (__16.z*v__11.z);
        __15.w = (__16.w*v__11.w);
      float4 __17;
        float4 __18;
        uint2 v__12 = *(uint2*)(k_tile_local_cast_12 + 0);
        ((float2*)(&__18))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[0]);
        ((float2*)(&__18))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[1]);
        float4 v__13 = *(float4*)(delta_local_cast_13 + 0);
        __17.x = (__18.x*v__13.x);
        __17.y = (__18.y*v__13.y);
        __17.z = (__18.z*v__13.z);
        __17.w = (__18.w*v__13.w);
      __14.x = (__15.x+__17.x);
      __14.y = (__15.y+__17.y);
      __14.z = (__15.z+__17.z);
      __14.w = (__15.w+__17.w);
    (reinterpret_cast<__nv_bfloat162*>(&__13))[0] = __float22bfloat162_rn(((float2*)(&__14))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__13))[1] = __float22bfloat162_rn(((float2*)(&__14))[1]);
    *(uint2*)(NewState_local_cast_10 + 0) = __13;
    *(uint2*)(NewState + (((((((int)blockIdx.y) * 393216) + (((int)blockIdx.x) * 16384)) + (i_12 * 512)) + (((int)threadIdx.x) * 4)) + 14336)) = *(uint2*)(NewState_local_cast_10 + 0);
  }
  ((float*)out_sq)[((int)threadIdx.x)] = (((float*)output)[((int)threadIdx.x)] * ((float*)output)[((int)threadIdx.x)]);
  out_sq_frag[0] = ((float*)out_sq)[((int)threadIdx.x)];
  var_buf_frag[0] = 0x0p+0f/*0.000000e+00*/;
  var_buf_frag[0] = (var_buf_frag[0] + out_sq_frag[0]);
  __syncthreads();
  var_buf_frag[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(var_buf_frag[0], (&(((float*)workspace)[0])));
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    ((float*)var_buf)[0] = var_buf_frag[0];
  }
  __syncthreads();
  float var_val = (((float*)var_buf)[0] / 0x1p+7f/*1.280000e+02*/);
  float rrms = rsqrtf((var_val + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  ((float*)norm_w)[((int)threadIdx.x)] = ((float)NormW[((int)threadIdx.x)]);
  ((float*)o_normed)[((int)threadIdx.x)] = ((((float*)output)[((int)threadIdx.x)] * rsqrtf((var_val + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/))) * ((float*)norm_w)[((int)threadIdx.x)]);
  ((float*)z_val)[((int)threadIdx.x)] = ((float)Z[(((((int)blockIdx.y) * 3072) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))]);
  float sig_z = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((float*)z_val)[((int)threadIdx.x)]))));
  __syncthreads();
  ((bfloat16_t*)final_output)[((int)threadIdx.x)] = ((bfloat16_t)((((float*)o_normed)[((int)threadIdx.x)] * sig_z) * ((float*)z_val)[((int)threadIdx.x)]));
  Output[(((((int)blockIdx.y) * 3072) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = ((bfloat16_t*)final_output)[((int)threadIdx.x)];
}

