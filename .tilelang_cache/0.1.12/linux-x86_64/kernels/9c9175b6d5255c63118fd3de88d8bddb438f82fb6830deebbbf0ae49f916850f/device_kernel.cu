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

extern "C" __global__ void main_kernel(bfloat16_t* __restrict__ K_Cache, const bfloat16_t* __restrict__ Key, const int* __restrict__ Slot_Mapping, bfloat16_t* __restrict__ V_Cache, const bfloat16_t* __restrict__ Value);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(bfloat16_t* __restrict__ K_Cache, const bfloat16_t* __restrict__ Key, const int* __restrict__ Slot_Mapping, bfloat16_t* __restrict__ V_Cache, const bfloat16_t* __restrict__ Value) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  void* slot_shared = ((void*)((char*)buf_dyn_shmem + 131072));
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((((((int)blockIdx.x) * 64) + (i * 2)) + (((int)threadIdx.x) >> 6)) < 205)) {
      condval = *(uint4*)(Key + (((((int)blockIdx.x) * 32768) + (i * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)K_shared) + ((i * 1024) + (((int)threadIdx.x) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_1;
    if (((((((int)blockIdx.x) * 64) + (i_1 * 2)) + (((int)threadIdx.x) >> 6)) < 205)) {
      condval_1 = *(uint4*)(Value + (((((int)blockIdx.x) * 32768) + (i_1 * 1024)) + (((int)threadIdx.x) * 8)));
    } else {
      condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)V_shared) + ((i_1 * 1024) + (((int)threadIdx.x) * 8))) = condval_1;
  }
  if (((int)threadIdx.x) < 64) {
    int condval_2;
    if ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < 205)) {
      condval_2 = Slot_Mapping[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))];
    } else {
      condval_2 = 0;
    }
    ((int*)slot_shared)[((int)threadIdx.x)] = condval_2;
  }
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    int slot = ((int*)slot_shared)[((i_2 * 2) + (((int)threadIdx.x) >> 6))];
    if (0 <= slot) {
      if (slot < 199776) {
        *(uint4*)(K_Cache + ((slot * 512) + ((((int)threadIdx.x) & 63) * 8))) = *(uint4*)(((bfloat16_t*)K_shared) + ((i_2 * 1024) + (((int)threadIdx.x) * 8)));
        *(uint4*)(V_Cache + ((slot * 512) + ((((int)threadIdx.x) & 63) * 8))) = *(uint4*)(((bfloat16_t*)V_shared) + ((i_2 * 1024) + (((int)threadIdx.x) * 8)));
      }
    }
  }
}

