#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(int64_t* __restrict__ out_idx, float* __restrict__ out_max, const int64_t* __restrict__ partial_idx, const float* __restrict__ partial_max);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(int64_t* __restrict__ out_idx, float* __restrict__ out_max, const int64_t* __restrict__ partial_idx, const float* __restrict__ partial_max) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* pidx_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* red_buf_staging = ((void*)((char*)buf_dyn_shmem + 0));
  void* red_result = ((void*)((char*)buf_dyn_shmem + 0));
  void* red_buf_staging_1 = ((void*)((char*)buf_dyn_shmem + 16));
  void* red_result_1 = ((void*)((char*)buf_dyn_shmem + 16));
  void* pmax_s = ((void*)((char*)buf_dyn_shmem + 16384));
  float local_max[4];
  float local_idx_f[4];
  float global_max[1];
  float global_min_idx_f[1];
  for (int i = 0; i < 4; ++i) {
    local_max[i] = -CUDART_INF_F;
    local_idx_f[i] = CUDART_INF_F;
  }
  for (int ko = 0; ko < 2; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      float condval;
      if ((((((ko * 256) + (i_1 * 16)) + (((int)threadIdx.x) >> 3)) < 485) && ((((int)threadIdx.x) & 3) < 1))) {
        condval = partial_max[((((ko * 512) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) + (((int)threadIdx.x) & 3))];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)pmax_s)[((i_1 * 128) + ((int)threadIdx.x))] = condval;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      int64_t condval_1;
      if ((((((ko * 256) + (i_2 * 16)) + (((int)threadIdx.x) >> 3)) < 485) && ((((int)threadIdx.x) & 3) < 1))) {
        condval_1 = partial_idx[((((ko * 512) + (i_2 * 32)) + (((int)threadIdx.x) >> 2)) + (((int)threadIdx.x) & 3))];
      } else {
        condval_1 = (int64_t)0;
      }
      ((int64_t*)pidx_s)[((i_2 * 128) + ((int)threadIdx.x))] = condval_1;
    }
    __syncthreads();
    for (int j = 0; j < 4; ++j) {
      if ((((ko * 256) + (j * 64)) + (((int)threadIdx.x) >> 1)) < 485) {
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          float val = ((float*)pmax_s)[(((j * 512) + (((int)threadIdx.x) * 4)) + i_3)];
          float idx_f = ((float)((int64_t*)pidx_s)[(((j * 512) + (((int)threadIdx.x) * 4)) + i_3)]);
          if (local_max[i_3] < val) {
            local_max[i_3] = val;
            local_idx_f[i_3] = idx_f;
          } else {
            if (val == local_max[i_3]) {
              if (idx_f < local_idx_f[i_3]) {
                local_idx_f[i_3] = idx_f;
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    float red_buf0[1];
    float t0[1];
    uint mask[1];
    float red_buf0_1[1];
    float t0_1[1];
    uint mask_1[1];
    red_buf0[0] = local_max[i_4];
    mask[0] = __activemask();
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
    red_buf0[0] = max(red_buf0[0], t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
    red_buf0[0] = max(red_buf0[0], t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
    red_buf0[0] = max(red_buf0[0], t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
    red_buf0[0] = max(red_buf0[0], t0[0]);
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
    red_buf0[0] = max(red_buf0[0], t0[0]);
    if ((((int)threadIdx.x) % 32) == 0) {
      ((float*)red_buf_staging)[(((int)threadIdx.x) >> 5)] = red_buf0[0];
    }
    __syncthreads();
    if (((int)threadIdx.x) < 4) {
      red_buf0_1[0] = ((float*)red_buf_staging)[((int)threadIdx.x)];
    }
    mask_1[0] = __activemask();
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
    red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
    if (((int)threadIdx.x) == 0) {
      ((float*)red_result)[0] = red_buf0_1[0];
    }
    __syncthreads();
    if (local_max[i_4] != ((float*)red_result)[0]) {
      local_idx_f[i_4] = CUDART_INF_F;
    }
    float red_buf0_2[1];
    float t0_2[1];
    uint mask_2[1];
    float red_buf0_3[1];
    float t0_3[1];
    uint mask_3[1];
    red_buf0_2[0] = local_idx_f[i_4];
    mask_2[0] = __activemask();
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 16, 32);
    red_buf0_2[0] = min(red_buf0_2[0], t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 8, 32);
    red_buf0_2[0] = min(red_buf0_2[0], t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 4, 32);
    red_buf0_2[0] = min(red_buf0_2[0], t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
    red_buf0_2[0] = min(red_buf0_2[0], t0_2[0]);
    t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
    red_buf0_2[0] = min(red_buf0_2[0], t0_2[0]);
    if ((((int)threadIdx.x) % 32) == 0) {
      ((float*)red_buf_staging_1)[(((int)threadIdx.x) >> 5)] = red_buf0_2[0];
    }
    __syncthreads();
    if (((int)threadIdx.x) < 4) {
      red_buf0_3[0] = ((float*)red_buf_staging_1)[((int)threadIdx.x)];
    }
    mask_3[0] = __activemask();
    t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
    red_buf0_3[0] = min(red_buf0_3[0], t0_3[0]);
    t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
    red_buf0_3[0] = min(red_buf0_3[0], t0_3[0]);
    if (((int)threadIdx.x) == 0) {
      ((float*)red_result_1)[0] = red_buf0_3[0];
    }
    __syncthreads();
    if (((int)threadIdx.x) == 0) {
      if (i_4 < 1) {
        out_idx[i_4] = ((int64_t)((int)((float*)red_result_1)[0]));
        out_max[i_4] = ((float*)red_result)[0];
      }
    }
  }
}

