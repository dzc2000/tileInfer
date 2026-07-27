// tilelang target: {"kind":"c","tag":"","keys":["cpu"]}
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#if defined(_MSC_VER)
#define TL_ALIGN(N) __declspec(align(N))
#else
#define TL_ALIGN(N) __attribute__((aligned(N)))
#endif
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
static void* main_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[14];
  void* stack_ffi_any = stack;
  if (!((num_args == 6))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 6", (long long)(num_args), (long long)(6));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t Q_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((Q_handle_type_index == 0) || (Q_handle_type_index == 4)) || (Q_handle_type_index == 7)) || (64 <= Q_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q expected pointer or tensor handle");
    return -1;
  }
  int32_t K_cache_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((K_cache_handle_type_index == 0) || (K_cache_handle_type_index == 4)) || (K_cache_handle_type_index == 7)) || (64 <= K_cache_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_cache expected pointer or tensor handle");
    return -1;
  }
  int32_t V_cache_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((V_cache_handle_type_index == 0) || (V_cache_handle_type_index == 4)) || (V_cache_handle_type_index == 7)) || (64 <= V_cache_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_cache expected pointer or tensor handle");
    return -1;
  }
  int32_t block_table_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((block_table_handle_type_index == 0) || (block_table_handle_type_index == 4)) || (block_table_handle_type_index == 7)) || (64 <= block_table_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input block_table expected pointer or tensor handle");
    return -1;
  }
  int32_t seqlen_kv_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((seqlen_kv_handle_type_index == 0) || (seqlen_kv_handle_type_index == 4)) || (seqlen_kv_handle_type_index == 7)) || (64 <= seqlen_kv_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input seqlen_kv expected pointer or tensor handle");
    return -1;
  }
  int32_t Output_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((Output_handle_type_index == 0) || (Output_handle_type_index == 4)) || (Output_handle_type_index == 7)) || (64 <= Output_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Output expected pointer or tensor handle");
    return -1;
  }
  void* Q_handle = ((Q_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* K_cache_handle = ((K_cache_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* V_cache_handle = ((V_cache_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* block_table_handle = ((block_table_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* seqlen_kv_handle = ((seqlen_kv_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* Output_handle = ((Output_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  bool main_Q_is_null = (Q_handle == NULL);
  if (!(!main_Q_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Q is expected to have non-NULL pointer");
    return -1;
  }
  bool main_K_cache_is_null = (K_cache_handle == NULL);
  if (!(!main_K_cache_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.K_cache is expected to have non-NULL pointer");
    return -1;
  }
  bool main_V_cache_is_null = (V_cache_handle == NULL);
  if (!(!main_V_cache_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.V_cache is expected to have non-NULL pointer");
    return -1;
  }
  bool main_block_table_is_null = (block_table_handle == NULL);
  if (!(!main_block_table_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.block_table is expected to have non-NULL pointer");
    return -1;
  }
  bool main_seqlen_kv_is_null = (seqlen_kv_handle == NULL);
  if (!(!main_seqlen_kv_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.seqlen_kv is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Output_is_null = (Output_handle == NULL);
  if (!(!main_Output_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Output is expected to have non-NULL pointer");
    return -1;
  }
  void* main_Q_shape = (((DLTensor*)Q_handle)[0].shape);
  void* main_K_cache_shape = (((DLTensor*)K_cache_handle)[0].shape);
  void* main_V_cache_shape = (((DLTensor*)V_cache_handle)[0].shape);
  void* main_block_table_shape = (((DLTensor*)block_table_handle)[0].shape);
  void* main_seqlen_kv_shape = (((DLTensor*)seqlen_kv_handle)[0].shape);
  void* main_Output_shape = (((DLTensor*)Output_handle)[0].shape);
  if (!(((((DLTensor*)Q_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q ndim mismatch, expected 3", (long long)((((DLTensor*)Q_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Q_strides = (((DLTensor*)Q_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)Q_handle)[0].device.device_id);
  void* Q = (((DLTensor*)Q_handle)[0].data);
  if (!(((((DLTensor*)K_cache_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache ndim mismatch, expected 3", (long long)((((DLTensor*)K_cache_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_K_cache_strides = (((DLTensor*)K_cache_handle)[0].strides);
  void* K_cache = (((DLTensor*)K_cache_handle)[0].data);
  if (!(((((DLTensor*)V_cache_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache ndim mismatch, expected 3", (long long)((((DLTensor*)V_cache_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_V_cache_strides = (((DLTensor*)V_cache_handle)[0].strides);
  void* V_cache = (((DLTensor*)V_cache_handle)[0].data);
  if (!(((((DLTensor*)block_table_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table ndim mismatch, expected 2", (long long)((((DLTensor*)block_table_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_block_table_strides = (((DLTensor*)block_table_handle)[0].strides);
  void* block_table = (((DLTensor*)block_table_handle)[0].data);
  if (!(((((DLTensor*)seqlen_kv_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input seqlen_kv ndim mismatch, expected 1", (long long)((((DLTensor*)seqlen_kv_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_seqlen_kv_strides = (((DLTensor*)seqlen_kv_handle)[0].strides);
  void* seqlen_kv = (((DLTensor*)seqlen_kv_handle)[0].data);
  if (!(((((DLTensor*)Output_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output ndim mismatch, expected 3", (long long)((((DLTensor*)Output_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Output_strides = (((DLTensor*)Output_handle)[0].strides);
  void* Output = (((DLTensor*)Output_handle)[0].data);
  if (!(((((((DLTensor*)Q_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Q_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Q_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Q_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Q_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Q_shape)[1]) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Q_shape)[1])), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Q_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Q_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_Q_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)main_Q_strides)[2]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_Q_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_Q_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q strides[2] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_Q_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_Q_strides)[1]);
  }
  if (!((condval_2 == 256))) {
    int32_t condval_3;
    if ((main_Q_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_Q_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q strides[1] violates packed ABI constraint", (long long)(condval_3), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((main_Q_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)main_Q_strides)[0]);
  }
  if (!((condval_4 == 3072))) {
    int32_t condval_5;
    if ((main_Q_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)main_Q_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q strides[0] violates packed ABI constraint", (long long)(condval_5), (long long)(3072));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Q_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Q_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Q_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q device_type mismatch, expected cuda", (long long)((((DLTensor*)Q_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Q == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)K_cache_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)K_cache_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)K_cache_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_cache dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_cache_shape)[0]) == 293536))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_cache_shape)[0])), (long long)(293536));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_cache_shape)[1]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_cache_shape)[1])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_cache_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_cache_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((main_K_cache_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)main_K_cache_strides)[2]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((main_K_cache_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)main_K_cache_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache strides[2] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((main_K_cache_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)main_K_cache_strides)[1]);
  }
  if (!((condval_8 == 256))) {
    int32_t condval_9;
    if ((main_K_cache_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)main_K_cache_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache strides[1] violates packed ABI constraint", (long long)(condval_9), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((main_K_cache_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)main_K_cache_strides)[0]);
  }
  if (!((condval_10 == 512))) {
    int32_t condval_11;
    if ((main_K_cache_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)main_K_cache_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache strides[0] violates packed ABI constraint", (long long)(condval_11), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)K_cache_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)K_cache_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_cache_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache device_id violates packed ABI constraint", (long long)((((DLTensor*)K_cache_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_cache_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_cache device_type mismatch, expected cuda", (long long)((((DLTensor*)K_cache_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(K_cache == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_cache data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)V_cache_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)V_cache_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)V_cache_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_cache dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_cache_shape)[0]) == 293536))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_cache_shape)[0])), (long long)(293536));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_cache_shape)[1]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_cache_shape)[1])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_cache_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_cache_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((main_V_cache_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)main_V_cache_strides)[2]);
  }
  if (!((condval_12 == 1))) {
    int32_t condval_13;
    if ((main_V_cache_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)main_V_cache_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache strides[2] violates packed ABI constraint", (long long)(condval_13), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((main_V_cache_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)main_V_cache_strides)[1]);
  }
  if (!((condval_14 == 256))) {
    int32_t condval_15;
    if ((main_V_cache_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)main_V_cache_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache strides[1] violates packed ABI constraint", (long long)(condval_15), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((main_V_cache_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)main_V_cache_strides)[0]);
  }
  if (!((condval_16 == 512))) {
    int32_t condval_17;
    if ((main_V_cache_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)main_V_cache_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache strides[0] violates packed ABI constraint", (long long)(condval_17), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)V_cache_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)V_cache_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_cache_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache device_id violates packed ABI constraint", (long long)((((DLTensor*)V_cache_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_cache_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_cache device_type mismatch, expected cuda", (long long)((((DLTensor*)V_cache_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(V_cache == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_cache data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)block_table_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)block_table_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)block_table_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input block_table dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_block_table_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_block_table_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_block_table_shape)[1]) == 18346))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_block_table_shape)[1])), (long long)(18346));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_18;
  if ((main_block_table_strides == NULL)) {
    condval_18 = 1;
  } else {
    condval_18 = ((int32_t)((int64_t*)main_block_table_strides)[1]);
  }
  if (!((condval_18 == 1))) {
    int32_t condval_19;
    if ((main_block_table_strides == NULL)) {
      condval_19 = 1;
    } else {
      condval_19 = ((int32_t)((int64_t*)main_block_table_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table strides[1] violates packed ABI constraint", (long long)(condval_19), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_20;
  if ((main_block_table_strides == NULL)) {
    condval_20 = 1;
  } else {
    condval_20 = ((int32_t)((int64_t*)main_block_table_strides)[0]);
  }
  if (!((condval_20 == 18346))) {
    int32_t condval_21;
    if ((main_block_table_strides == NULL)) {
      condval_21 = 1;
    } else {
      condval_21 = ((int32_t)((int64_t*)main_block_table_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table strides[0] violates packed ABI constraint", (long long)(condval_21), (long long)(18346));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)block_table_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)block_table_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)block_table_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table device_id violates packed ABI constraint", (long long)((((DLTensor*)block_table_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)block_table_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input block_table device_type mismatch, expected cuda", (long long)((((DLTensor*)block_table_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(block_table == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input block_table data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)seqlen_kv_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)seqlen_kv_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)seqlen_kv_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input seqlen_kv dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_seqlen_kv_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input seqlen_kv shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_seqlen_kv_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((main_seqlen_kv_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)main_seqlen_kv_strides)[0]);
  }
  if (!((condval_22 == 1))) {
    int32_t condval_23;
    if ((main_seqlen_kv_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)main_seqlen_kv_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input seqlen_kv strides[0] violates packed ABI constraint", (long long)(condval_23), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)seqlen_kv_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input seqlen_kv byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)seqlen_kv_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)seqlen_kv_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input seqlen_kv device_id violates packed ABI constraint", (long long)((((DLTensor*)seqlen_kv_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)seqlen_kv_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input seqlen_kv device_type mismatch, expected cuda", (long long)((((DLTensor*)seqlen_kv_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(seqlen_kv == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input seqlen_kv data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Output_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Output_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Output_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Output dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Output_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Output_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Output_shape)[1]) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Output_shape)[1])), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Output_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Output_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((main_Output_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)main_Output_strides)[2]);
  }
  if (!((condval_24 == 1))) {
    int32_t condval_25;
    if ((main_Output_strides == NULL)) {
      condval_25 = 1;
    } else {
      condval_25 = ((int32_t)((int64_t*)main_Output_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output strides[2] violates packed ABI constraint", (long long)(condval_25), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((main_Output_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)main_Output_strides)[1]);
  }
  if (!((condval_26 == 256))) {
    int32_t condval_27;
    if ((main_Output_strides == NULL)) {
      condval_27 = 1;
    } else {
      condval_27 = ((int32_t)((int64_t*)main_Output_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output strides[1] violates packed ABI constraint", (long long)(condval_27), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_28;
  if ((main_Output_strides == NULL)) {
    condval_28 = 1;
  } else {
    condval_28 = ((int32_t)((int64_t*)main_Output_strides)[0]);
  }
  if (!((condval_28 == 3072))) {
    int32_t condval_29;
    if ((main_Output_strides == NULL)) {
      condval_29 = 1;
    } else {
      condval_29 = ((int32_t)((int64_t*)main_Output_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output strides[0] violates packed ABI constraint", (long long)(condval_29), (long long)(3072));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Output_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Output_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output device_id violates packed ABI constraint", (long long)((((DLTensor*)Output_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output device_type mismatch, expected cuda", (long long)((((DLTensor*)Output_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Output == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Output data pointer is NULL");
    return -1;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_1;
  result_1.type_index = kTVMFFINone;
  result_1.zero_padding = 0;
  result_1.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_1) != 0) {
    return -1;
  }
  if (K_cache == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = K_cache;
  if (Output == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = Output;
  if (Q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = Q;
  if (V_cache == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = V_cache;
  if (block_table == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = block_table;
  if (seqlen_kv == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = seqlen_kv;
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)77824);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 13, &result_2) != 0) {
    return -1;
  }
  return 0;
}

