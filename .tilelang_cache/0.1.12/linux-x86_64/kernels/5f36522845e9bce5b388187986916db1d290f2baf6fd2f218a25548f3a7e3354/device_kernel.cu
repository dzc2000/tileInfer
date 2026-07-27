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
  void* output = ((void*)((char*)buf_dyn_shmem + 528));
  void* q_full = ((void*)((char*)buf_dyn_shmem + 528));
  void* k_full = ((void*)((char*)buf_dyn_shmem + 784));
  void* norm_w = ((void*)((char*)buf_dyn_shmem + 1040));
  void* out_sq = ((void*)((char*)buf_dyn_shmem + 1040));
  void* q_tile = ((void*)((char*)buf_dyn_shmem + 1040));
  void* v = ((void*)((char*)buf_dyn_shmem + 1040));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 1040));
  void* S_read = ((void*)((char*)buf_dyn_shmem + 1296));
  void* accumulated = ((void*)((char*)buf_dyn_shmem + 1296));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 1296));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 1296));
  void* o_normed = ((void*)((char*)buf_dyn_shmem + 1552));
  void* delta = ((void*)((char*)buf_dyn_shmem + 1808));
  void* S_read2 = ((void*)((char*)buf_dyn_shmem + 2320));
  void* S_decayed = ((void*)((char*)buf_dyn_shmem + 13584));
  void* S_updated = ((void*)((char*)buf_dyn_shmem + 14608));
  void* partial_accum = ((void*)((char*)buf_dyn_shmem + 21776));
  void* k_tile = ((void*)((char*)buf_dyn_shmem + 25872));
  void* partial_output = ((void*)((char*)buf_dyn_shmem + 25904));
  float pow_frag[1];
  float pow_sum_frag[1];
  float dot_k[1];
  bfloat16_t S_read_local_cast_1[4];
  float S_decayed_local_cast[4];
  float S_decayed_local_cast_3[4];
  bfloat16_t NewState_local_cast_2[4];
  bfloat16_t S_read_local_cast_5[4];
  float S_decayed_local_cast_4[4];
  float S_decayed_local_cast_7[4];
  bfloat16_t NewState_local_cast_6[4];
  bfloat16_t S_read_local_cast_9[4];
  float S_decayed_local_cast_8[4];
  float S_decayed_local_cast_11[4];
  bfloat16_t NewState_local_cast_10[4];
  bfloat16_t S_read_local_cast_13[4];
  float S_decayed_local_cast_12[4];
  float S_decayed_local_cast_15[4];
  bfloat16_t NewState_local_cast_14[4];
  float dot_q[1];
  bfloat16_t S_read2_local_cast_17[4];
  bfloat16_t k_tile_local_cast_18[4];
  float delta_local_cast_19[4];
  float S_updated_local_cast_16[4];
  float S_updated_local_cast_21[4];
  bfloat16_t NewState_local_cast_20[4];
  bfloat16_t S_read2_local_cast_23[4];
  bfloat16_t k_tile_local_cast_24[4];
  float delta_local_cast_25[4];
  float S_updated_local_cast_22[4];
  float S_updated_local_cast_27[4];
  bfloat16_t NewState_local_cast_26[4];
  bfloat16_t S_read2_local_cast_29[4];
  bfloat16_t k_tile_local_cast_30[4];
  float delta_local_cast_31[4];
  float S_updated_local_cast_28[4];
  float S_updated_local_cast_33[4];
  bfloat16_t NewState_local_cast_32[4];
  bfloat16_t S_read2_local_cast_35[4];
  bfloat16_t k_tile_local_cast_36[4];
  float delta_local_cast_37[4];
  float S_updated_local_cast_34[4];
  float S_updated_local_cast_39[4];
  bfloat16_t NewState_local_cast_38[4];
  float out_sq_frag[1];
  float var_buf_frag[1];
  ((bfloat16_t*)q_full)[((int)threadIdx.x)] = QKV[(((((int)blockIdx.x) / 3) * 128) + ((int)threadIdx.x))];
  ((bfloat16_t*)k_full)[((int)threadIdx.x)] = QKV[((((((int)blockIdx.x) / 3) * 128) + ((int)threadIdx.x)) + 1024)];
  ((bfloat16_t*)v)[((int)threadIdx.x)] = QKV[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 2048)];
  pow_frag[0] = (((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]) * ((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]));
  pow_sum_frag[0] = 0x0p+0f/*0.000000e+00*/;
  pow_sum_frag[0] = (pow_sum_frag[0] + pow_frag[0]);
  __syncthreads();
  pow_sum_frag[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(pow_sum_frag[0], (&(((float*)workspace_1)[0])));
  float q_norm = max(sqrtf(pow_sum_frag[0]), 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  __syncthreads();
  ((bfloat16_t*)q_full)[((int)threadIdx.x)] = ((bfloat16_t)((((float)((bfloat16_t*)q_full)[((int)threadIdx.x)]) / q_norm) * 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/));
  pow_frag[0] = (((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]) * ((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]));
  pow_sum_frag[0] = 0x0p+0f/*0.000000e+00*/;
  pow_sum_frag[0] = (pow_sum_frag[0] + pow_frag[0]);
  __syncthreads();
  pow_sum_frag[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(pow_sum_frag[0], (&(((float*)workspace_2)[0])));
  float k_norm = max(sqrtf(pow_sum_frag[0]), 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/);
  __syncthreads();
  ((bfloat16_t*)k_full)[((int)threadIdx.x)] = ((bfloat16_t)(((float)((bfloat16_t*)k_full)[((int)threadIdx.x)]) / k_norm));
  float alpha_val = ((float)Alpha[((int)blockIdx.x)]);
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
  float beta = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((float)BetaRaw[((int)blockIdx.x)])))));
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_read)[((i * 1024) + (((int)threadIdx.x) * 8))])), (&(State[(((((int)blockIdx.x) * 16384) + (i * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_read)[(((i_1 * 1024) + (((int)threadIdx.x) * 8)) + 2048)])), (&(State[((((((int)blockIdx.x) * 16384) + (i_1 * 1024)) + (((int)threadIdx.x) * 8)) + 2048)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_read)[(((i_2 * 1024) + (((int)threadIdx.x) * 8)) + 4096)])), (&(State[((((((int)blockIdx.x) * 16384) + (i_2 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
  }
  tl::cp_async_commit();
  for (int r_start = 0; r_start < 5; ++r_start) {
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[((r_start * 16) + ((int)threadIdx.x))];
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      *(uint2*)(S_read_local_cast_1 + 0) = *(uint2*)(((bfloat16_t*)S_read) + ((((r_start % 3) * 2048) + (i_3 * 512)) + (((int)threadIdx.x) * 4)));
      float4 __1;
        float4 __2;
        uint2 v_ = *(uint2*)(S_read_local_cast_1 + 0);
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
      *(float4*)(S_decayed_local_cast + 0) = __1;
      *(float4*)(((float*)S_decayed) + ((i_3 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_local_cast + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)S_read)[((((r_start % 3) * 2048) + (i_4 * 1024)) + (((int)threadIdx.x) * 8))])), (&(State[(((((((int)blockIdx.x) * 16384) + (r_start * 2048)) + (i_4 * 1024)) + (((int)threadIdx.x) * 8)) + 6144)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      *(float4*)(S_decayed_local_cast_3 + 0) = *(float4*)(((float*)S_decayed) + ((i_5 * 512) + (((int)threadIdx.x) * 4)));
      uint2 __3;
      float4 v__2 = *(float4*)(S_decayed_local_cast_3 + 0);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(NewState_local_cast_2 + 0) = __3;
      *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (r_start * 2048)) + (i_5 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(NewState_local_cast_2 + 0);
    }
    dot_k[0] = 0x0p+0f/*0.000000e+00*/;
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      dot_k[0] = (dot_k[0] + (((float*)S_decayed)[((i_6 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)k_tile)[i_6])));
    }
    ((float*)partial_accum)[((r_start * 128) + ((int)threadIdx.x))] = dot_k[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 80)];
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    *(uint2*)(S_read_local_cast_5 + 0) = *(uint2*)(((bfloat16_t*)S_read) + (((i_7 * 512) + (((int)threadIdx.x) * 4)) + 4096));
    float4 __4;
      float4 __5;
      uint2 v__3 = *(uint2*)(S_read_local_cast_5 + 0);
      ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
      ((float2*)(&__5))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
      float condval_4;
      if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
        condval_4 = (alpha_val + dt_bias_val);
      } else {
        condval_4 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
      }
      float4 v__4 = make_float4(expf((neg_a_exp_val * condval_4)), expf((neg_a_exp_val * condval_4)), expf((neg_a_exp_val * condval_4)), expf((neg_a_exp_val * condval_4)));
      __4.x = (__5.x*v__4.x);
      __4.y = (__5.y*v__4.y);
      __4.z = (__5.z*v__4.z);
      __4.w = (__5.w*v__4.w);
    *(float4*)(S_decayed_local_cast_4 + 0) = __4;
    *(float4*)(((float*)S_decayed) + ((i_7 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_local_cast_4 + 0);
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    *(float4*)(S_decayed_local_cast_7 + 0) = *(float4*)(((float*)S_decayed) + ((i_8 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __6;
    float4 v__5 = *(float4*)(S_decayed_local_cast_7 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__5))[1]);
    *(uint2*)(NewState_local_cast_6 + 0) = __6;
    *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (i_8 * 512)) + (((int)threadIdx.x) * 4)) + 10240)) = *(uint2*)(NewState_local_cast_6 + 0);
  }
  dot_k[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_9 = 0; i_9 < 16; ++i_9) {
    dot_k[0] = (dot_k[0] + (((float*)S_decayed)[((i_9 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)k_tile)[i_9])));
  }
  ((float*)partial_accum)[(((int)threadIdx.x) + 640)] = dot_k[0];
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 96)];
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    *(uint2*)(S_read_local_cast_9 + 0) = *(uint2*)(((bfloat16_t*)S_read) + ((i_10 * 512) + (((int)threadIdx.x) * 4)));
    float4 __7;
      float4 __8;
      uint2 v__6 = *(uint2*)(S_read_local_cast_9 + 0);
      ((float2*)(&__8))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[0]);
      ((float2*)(&__8))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__6))[1]);
      float condval_5;
      if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
        condval_5 = (alpha_val + dt_bias_val);
      } else {
        condval_5 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
      }
      float4 v__7 = make_float4(expf((neg_a_exp_val * condval_5)), expf((neg_a_exp_val * condval_5)), expf((neg_a_exp_val * condval_5)), expf((neg_a_exp_val * condval_5)));
      __7.x = (__8.x*v__7.x);
      __7.y = (__8.y*v__7.y);
      __7.z = (__8.z*v__7.z);
      __7.w = (__8.w*v__7.w);
    *(float4*)(S_decayed_local_cast_8 + 0) = __7;
    *(float4*)(((float*)S_decayed) + ((i_10 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_local_cast_8 + 0);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 4; ++i_11) {
    *(float4*)(S_decayed_local_cast_11 + 0) = *(float4*)(((float*)S_decayed) + ((i_11 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __9;
    float4 v__8 = *(float4*)(S_decayed_local_cast_11 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__9))[0] = __float22bfloat162_rn(((float2*)(&v__8))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__9))[1] = __float22bfloat162_rn(((float2*)(&v__8))[1]);
    *(uint2*)(NewState_local_cast_10 + 0) = __9;
    *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (i_11 * 512)) + (((int)threadIdx.x) * 4)) + 12288)) = *(uint2*)(NewState_local_cast_10 + 0);
  }
  dot_k[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    dot_k[0] = (dot_k[0] + (((float*)S_decayed)[((i_12 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)k_tile)[i_12])));
  }
  ((float*)partial_accum)[(((int)threadIdx.x) + 768)] = dot_k[0];
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 112)];
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_13 = 0; i_13 < 4; ++i_13) {
    *(uint2*)(S_read_local_cast_13 + 0) = *(uint2*)(((bfloat16_t*)S_read) + (((i_13 * 512) + (((int)threadIdx.x) * 4)) + 2048));
    float4 __10;
      float4 __11;
      uint2 v__9 = *(uint2*)(S_read_local_cast_13 + 0);
      ((float2*)(&__11))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[0]);
      ((float2*)(&__11))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__9))[1]);
      float condval_6;
      if ((0x1.4p+4f/*2.000000e+01*/ < (alpha_val + dt_bias_val))) {
        condval_6 = (alpha_val + dt_bias_val);
      } else {
        condval_6 = logf((0x1p+0f/*1.000000e+00*/ + expf((alpha_val + dt_bias_val))));
      }
      float4 v__10 = make_float4(expf((neg_a_exp_val * condval_6)), expf((neg_a_exp_val * condval_6)), expf((neg_a_exp_val * condval_6)), expf((neg_a_exp_val * condval_6)));
      __10.x = (__11.x*v__10.x);
      __10.y = (__11.y*v__10.y);
      __10.z = (__11.z*v__10.z);
      __10.w = (__11.w*v__10.w);
    *(float4*)(S_decayed_local_cast_12 + 0) = __10;
    *(float4*)(((float*)S_decayed) + ((i_13 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_decayed_local_cast_12 + 0);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 4; ++i_14) {
    *(float4*)(S_decayed_local_cast_15 + 0) = *(float4*)(((float*)S_decayed) + ((i_14 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __12;
    float4 v__11 = *(float4*)(S_decayed_local_cast_15 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__12))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__12))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
    *(uint2*)(NewState_local_cast_14 + 0) = __12;
    *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (i_14 * 512)) + (((int)threadIdx.x) * 4)) + 14336)) = *(uint2*)(NewState_local_cast_14 + 0);
  }
  dot_k[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_15 = 0; i_15 < 16; ++i_15) {
    dot_k[0] = (dot_k[0] + (((float*)S_decayed)[((i_15 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)k_tile)[i_15])));
  }
  ((float*)partial_accum)[(((int)threadIdx.x) + 896)] = dot_k[0];
  ((float*)accumulated)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int t = 0; t < 8; ++t) {
    ((float*)accumulated)[((int)threadIdx.x)] = (((float*)accumulated)[((int)threadIdx.x)] + ((float*)partial_accum)[((t * 128) + ((int)threadIdx.x))]);
  }
  __syncthreads();
  ((float*)delta)[((int)threadIdx.x)] = (beta * (((float)((bfloat16_t*)v)[((int)threadIdx.x)]) - ((float*)accumulated)[((int)threadIdx.x)]));
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_read2)[((i_16 * 1024) + (((int)threadIdx.x) * 8))])), (&(NewState[(((((int)blockIdx.x) * 16384) + (i_16 * 1024)) + (((int)threadIdx.x) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_read2)[(((i_17 * 1024) + (((int)threadIdx.x) * 8)) + 2048)])), (&(NewState[((((((int)blockIdx.x) * 16384) + (i_17 * 1024)) + (((int)threadIdx.x) * 8)) + 2048)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)S_read2)[(((i_18 * 1024) + (((int)threadIdx.x) * 8)) + 4096)])), (&(NewState[((((((int)blockIdx.x) * 16384) + (i_18 * 1024)) + (((int)threadIdx.x) * 8)) + 4096)])));
  }
  tl::cp_async_commit();
  __syncthreads();
  for (int r_start_1 = 0; r_start_1 < 5; ++r_start_1) {
    if (((int)threadIdx.x) < 16) {
      ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[((r_start_1 * 16) + ((int)threadIdx.x))];
    }
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[((r_start_1 * 16) + ((int)threadIdx.x))];
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 4; ++i_19) {
      *(uint2*)(S_read2_local_cast_17 + 0) = *(uint2*)(((bfloat16_t*)S_read2) + ((((r_start_1 % 3) * 2048) + (i_19 * 512)) + (((int)threadIdx.x) * 4)));
      *(uint2*)(k_tile_local_cast_18 + 0) = make_uint2(__pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_19 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_19 * 4) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_19 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_19 * 4) + (((int)threadIdx.x) >> 5))]));
      *(float4*)(delta_local_cast_19 + 0) = *(float4*)(((float*)delta) + ((((int)threadIdx.x) & 31) * 4));
      float4 __13;
        float4 __14;
        uint2 v__12 = *(uint2*)(S_read2_local_cast_17 + 0);
        ((float2*)(&__14))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[0]);
        ((float2*)(&__14))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__12))[1]);
        float4 __15;
          float4 __16;
          uint2 v__13 = *(uint2*)(k_tile_local_cast_18 + 0);
          ((float2*)(&__16))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__13))[0]);
          ((float2*)(&__16))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__13))[1]);
          float4 v__14 = *(float4*)(delta_local_cast_19 + 0);
          __15.x = (__16.x*v__14.x);
          __15.y = (__16.y*v__14.y);
          __15.z = (__16.z*v__14.z);
          __15.w = (__16.w*v__14.w);
        __13.x = (__14.x+__15.x);
        __13.y = (__14.y+__15.y);
        __13.z = (__14.z+__15.z);
        __13.w = (__14.w+__15.w);
      *(float4*)(S_updated_local_cast_16 + 0) = __13;
      *(float4*)(((float*)S_updated) + ((i_19 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_updated_local_cast_16 + 0);
    }
    __syncthreads();
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)S_read2)[((((r_start_1 % 3) * 2048) + (i_20 * 1024)) + (((int)threadIdx.x) * 8))])), (&(NewState[(((((((int)blockIdx.x) * 16384) + (r_start_1 * 2048)) + (i_20 * 1024)) + (((int)threadIdx.x) * 8)) + 6144)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    #pragma unroll
    for (int i_21 = 0; i_21 < 4; ++i_21) {
      *(float4*)(S_updated_local_cast_21 + 0) = *(float4*)(((float*)S_updated) + ((i_21 * 512) + (((int)threadIdx.x) * 4)));
      uint2 __17;
      float4 v__15 = *(float4*)(S_updated_local_cast_21 + 0);
      (reinterpret_cast<__nv_bfloat162*>(&__17))[0] = __float22bfloat162_rn(((float2*)(&v__15))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__17))[1] = __float22bfloat162_rn(((float2*)(&v__15))[1]);
      *(uint2*)(NewState_local_cast_20 + 0) = __17;
      *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (r_start_1 * 2048)) + (i_21 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(NewState_local_cast_20 + 0);
    }
    dot_q[0] = 0x0p+0f/*0.000000e+00*/;
    for (int i_22 = 0; i_22 < 16; ++i_22) {
      dot_q[0] = (dot_q[0] + (((float*)S_updated)[((i_22 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)q_tile)[i_22])));
    }
    ((float*)partial_output)[((r_start_1 * 128) + ((int)threadIdx.x))] = dot_q[0];
  }
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 80)];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[(((int)threadIdx.x) + 80)];
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 4; ++i_23) {
    *(uint2*)(S_read2_local_cast_23 + 0) = *(uint2*)(((bfloat16_t*)S_read2) + (((i_23 * 512) + (((int)threadIdx.x) * 4)) + 4096));
    *(uint2*)(k_tile_local_cast_24 + 0) = make_uint2(__pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_23 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_23 * 4) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_23 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_23 * 4) + (((int)threadIdx.x) >> 5))]));
    *(float4*)(delta_local_cast_25 + 0) = *(float4*)(((float*)delta) + ((((int)threadIdx.x) & 31) * 4));
    float4 __18;
      float4 __19;
      uint2 v__16 = *(uint2*)(S_read2_local_cast_23 + 0);
      ((float2*)(&__19))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__16))[0]);
      ((float2*)(&__19))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__16))[1]);
      float4 __20;
        float4 __21;
        uint2 v__17 = *(uint2*)(k_tile_local_cast_24 + 0);
        ((float2*)(&__21))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__17))[0]);
        ((float2*)(&__21))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__17))[1]);
        float4 v__18 = *(float4*)(delta_local_cast_25 + 0);
        __20.x = (__21.x*v__18.x);
        __20.y = (__21.y*v__18.y);
        __20.z = (__21.z*v__18.z);
        __20.w = (__21.w*v__18.w);
      __18.x = (__19.x+__20.x);
      __18.y = (__19.y+__20.y);
      __18.z = (__19.z+__20.z);
      __18.w = (__19.w+__20.w);
    *(float4*)(S_updated_local_cast_22 + 0) = __18;
    *(float4*)(((float*)S_updated) + ((i_23 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_updated_local_cast_22 + 0);
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 4; ++i_24) {
    *(float4*)(S_updated_local_cast_27 + 0) = *(float4*)(((float*)S_updated) + ((i_24 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __22;
    float4 v__19 = *(float4*)(S_updated_local_cast_27 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__22))[0] = __float22bfloat162_rn(((float2*)(&v__19))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__22))[1] = __float22bfloat162_rn(((float2*)(&v__19))[1]);
    *(uint2*)(NewState_local_cast_26 + 0) = __22;
    *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (i_24 * 512)) + (((int)threadIdx.x) * 4)) + 10240)) = *(uint2*)(NewState_local_cast_26 + 0);
  }
  dot_q[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_25 = 0; i_25 < 16; ++i_25) {
    dot_q[0] = (dot_q[0] + (((float*)S_updated)[((i_25 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)q_tile)[i_25])));
  }
  ((float*)partial_output)[(((int)threadIdx.x) + 640)] = dot_q[0];
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 96)];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[(((int)threadIdx.x) + 96)];
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 4; ++i_26) {
    *(uint2*)(S_read2_local_cast_29 + 0) = *(uint2*)(((bfloat16_t*)S_read2) + ((i_26 * 512) + (((int)threadIdx.x) * 4)));
    *(uint2*)(k_tile_local_cast_30 + 0) = make_uint2(__pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_26 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_26 * 4) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_26 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_26 * 4) + (((int)threadIdx.x) >> 5))]));
    *(float4*)(delta_local_cast_31 + 0) = *(float4*)(((float*)delta) + ((((int)threadIdx.x) & 31) * 4));
    float4 __23;
      float4 __24;
      uint2 v__20 = *(uint2*)(S_read2_local_cast_29 + 0);
      ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__20))[0]);
      ((float2*)(&__24))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__20))[1]);
      float4 __25;
        float4 __26;
        uint2 v__21 = *(uint2*)(k_tile_local_cast_30 + 0);
        ((float2*)(&__26))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__21))[0]);
        ((float2*)(&__26))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__21))[1]);
        float4 v__22 = *(float4*)(delta_local_cast_31 + 0);
        __25.x = (__26.x*v__22.x);
        __25.y = (__26.y*v__22.y);
        __25.z = (__26.z*v__22.z);
        __25.w = (__26.w*v__22.w);
      __23.x = (__24.x+__25.x);
      __23.y = (__24.y+__25.y);
      __23.z = (__24.z+__25.z);
      __23.w = (__24.w+__25.w);
    *(float4*)(S_updated_local_cast_28 + 0) = __23;
    *(float4*)(((float*)S_updated) + ((i_26 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_updated_local_cast_28 + 0);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 4; ++i_27) {
    *(float4*)(S_updated_local_cast_33 + 0) = *(float4*)(((float*)S_updated) + ((i_27 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __27;
    float4 v__23 = *(float4*)(S_updated_local_cast_33 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__27))[0] = __float22bfloat162_rn(((float2*)(&v__23))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__27))[1] = __float22bfloat162_rn(((float2*)(&v__23))[1]);
    *(uint2*)(NewState_local_cast_32 + 0) = __27;
    *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (i_27 * 512)) + (((int)threadIdx.x) * 4)) + 12288)) = *(uint2*)(NewState_local_cast_32 + 0);
  }
  dot_q[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_28 = 0; i_28 < 16; ++i_28) {
    dot_q[0] = (dot_q[0] + (((float*)S_updated)[((i_28 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)q_tile)[i_28])));
  }
  ((float*)partial_output)[(((int)threadIdx.x) + 768)] = dot_q[0];
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)k_tile)[((int)threadIdx.x)] = ((bfloat16_t*)k_full)[(((int)threadIdx.x) + 112)];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((bfloat16_t*)q_tile)[((int)threadIdx.x)] = ((bfloat16_t*)q_full)[(((int)threadIdx.x) + 112)];
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 4; ++i_29) {
    *(uint2*)(S_read2_local_cast_35 + 0) = *(uint2*)(((bfloat16_t*)S_read2) + (((i_29 * 512) + (((int)threadIdx.x) * 4)) + 2048));
    *(uint2*)(k_tile_local_cast_36 + 0) = make_uint2(__pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_29 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_29 * 4) + (((int)threadIdx.x) >> 5))]), __pack_nv_bfloat162(((bfloat16_t*)k_tile)[((i_29 * 4) + (((int)threadIdx.x) >> 5))], ((bfloat16_t*)k_tile)[((i_29 * 4) + (((int)threadIdx.x) >> 5))]));
    *(float4*)(delta_local_cast_37 + 0) = *(float4*)(((float*)delta) + ((((int)threadIdx.x) & 31) * 4));
    float4 __28;
      float4 __29;
      uint2 v__24 = *(uint2*)(S_read2_local_cast_35 + 0);
      ((float2*)(&__29))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__24))[0]);
      ((float2*)(&__29))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__24))[1]);
      float4 __30;
        float4 __31;
        uint2 v__25 = *(uint2*)(k_tile_local_cast_36 + 0);
        ((float2*)(&__31))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__25))[0]);
        ((float2*)(&__31))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__25))[1]);
        float4 v__26 = *(float4*)(delta_local_cast_37 + 0);
        __30.x = (__31.x*v__26.x);
        __30.y = (__31.y*v__26.y);
        __30.z = (__31.z*v__26.z);
        __30.w = (__31.w*v__26.w);
      __28.x = (__29.x+__30.x);
      __28.y = (__29.y+__30.y);
      __28.z = (__29.z+__30.z);
      __28.w = (__29.w+__30.w);
    *(float4*)(S_updated_local_cast_34 + 0) = __28;
    *(float4*)(((float*)S_updated) + ((i_29 * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(S_updated_local_cast_34 + 0);
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 4; ++i_30) {
    *(float4*)(S_updated_local_cast_39 + 0) = *(float4*)(((float*)S_updated) + ((i_30 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __32;
    float4 v__27 = *(float4*)(S_updated_local_cast_39 + 0);
    (reinterpret_cast<__nv_bfloat162*>(&__32))[0] = __float22bfloat162_rn(((float2*)(&v__27))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__32))[1] = __float22bfloat162_rn(((float2*)(&v__27))[1]);
    *(uint2*)(NewState_local_cast_38 + 0) = __32;
    *(uint2*)(NewState + ((((((int)blockIdx.x) * 16384) + (i_30 * 512)) + (((int)threadIdx.x) * 4)) + 14336)) = *(uint2*)(NewState_local_cast_38 + 0);
  }
  dot_q[0] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int i_31 = 0; i_31 < 16; ++i_31) {
    dot_q[0] = (dot_q[0] + (((float*)S_updated)[((i_31 * 128) + ((int)threadIdx.x))] * ((float)((bfloat16_t*)q_tile)[i_31])));
  }
  ((float*)partial_output)[(((int)threadIdx.x) + 896)] = dot_q[0];
  ((float*)output)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  __syncthreads();
  for (int t_1 = 0; t_1 < 8; ++t_1) {
    ((float*)output)[((int)threadIdx.x)] = (((float*)output)[((int)threadIdx.x)] + ((float*)partial_output)[((t_1 * 128) + ((int)threadIdx.x))]);
  }
  __syncthreads();
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
  ((float*)z_val)[((int)threadIdx.x)] = ((float)Z[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  float sig_z = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + expf((0x0p+0f/*0.000000e+00*/ - ((float*)z_val)[((int)threadIdx.x)]))));
  __syncthreads();
  ((bfloat16_t*)final_output)[((int)threadIdx.x)] = ((bfloat16_t)((((float*)o_normed)[((int)threadIdx.x)] * sig_z) * ((float*)z_val)[((int)threadIdx.x)]));
  Output[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((bfloat16_t*)final_output)[((int)threadIdx.x)];
}

