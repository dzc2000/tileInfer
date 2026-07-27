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
  TL_ALIGN(128) TVMFFIAny stack[15];
  void* stack_ffi_any = stack;
  if (!((num_args == 7))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 7", (long long)(num_args), (long long)(7));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t Q_unpad_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((Q_unpad_handle_type_index == 0) || (Q_unpad_handle_type_index == 4)) || (Q_unpad_handle_type_index == 7)) || (64 <= Q_unpad_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q_unpad expected pointer or tensor handle");
    return -1;
  }
  int32_t K_unpad_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((K_unpad_handle_type_index == 0) || (K_unpad_handle_type_index == 4)) || (K_unpad_handle_type_index == 7)) || (64 <= K_unpad_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_unpad expected pointer or tensor handle");
    return -1;
  }
  int32_t V_unpad_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((V_unpad_handle_type_index == 0) || (V_unpad_handle_type_index == 4)) || (V_unpad_handle_type_index == 7)) || (64 <= V_unpad_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_unpad expected pointer or tensor handle");
    return -1;
  }
  int32_t cu_seqlens_q_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((cu_seqlens_q_handle_type_index == 0) || (cu_seqlens_q_handle_type_index == 4)) || (cu_seqlens_q_handle_type_index == 7)) || (64 <= cu_seqlens_q_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input cu_seqlens_q expected pointer or tensor handle");
    return -1;
  }
  int32_t cu_seqlens_k_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((cu_seqlens_k_handle_type_index == 0) || (cu_seqlens_k_handle_type_index == 4)) || (cu_seqlens_k_handle_type_index == 7)) || (64 <= cu_seqlens_k_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input cu_seqlens_k expected pointer or tensor handle");
    return -1;
  }
  int32_t max_seqlen_q_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((max_seqlen_q_type_index == 1) || (max_seqlen_q_type_index == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main scalar max_seqlen_q expected integer");
    return -1;
  }
  int32_t Output_unpad_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((Output_unpad_handle_type_index == 0) || (Output_unpad_handle_type_index == 4)) || (Output_unpad_handle_type_index == 7)) || (64 <= Output_unpad_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Output_unpad expected pointer or tensor handle");
    return -1;
  }
  void* Q_unpad_handle = ((Q_unpad_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* K_unpad_handle = ((K_unpad_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* V_unpad_handle = ((V_unpad_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* cu_seqlens_q_handle = ((cu_seqlens_q_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* cu_seqlens_k_handle = ((cu_seqlens_k_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  int32_t max_seqlen_q = ((int32_t)(((TVMFFIAny*)args)[5].v_int64));
  void* Output_unpad_handle = ((Output_unpad_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  bool main_Q_unpad_is_null = (Q_unpad_handle == NULL);
  if (!(!main_Q_unpad_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Q_unpad is expected to have non-NULL pointer");
    return -1;
  }
  bool main_K_unpad_is_null = (K_unpad_handle == NULL);
  if (!(!main_K_unpad_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.K_unpad is expected to have non-NULL pointer");
    return -1;
  }
  bool main_V_unpad_is_null = (V_unpad_handle == NULL);
  if (!(!main_V_unpad_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.V_unpad is expected to have non-NULL pointer");
    return -1;
  }
  bool main_cu_seqlens_q_is_null = (cu_seqlens_q_handle == NULL);
  if (!(!main_cu_seqlens_q_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.cu_seqlens_q is expected to have non-NULL pointer");
    return -1;
  }
  bool main_cu_seqlens_k_is_null = (cu_seqlens_k_handle == NULL);
  if (!(!main_cu_seqlens_k_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.cu_seqlens_k is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Output_unpad_is_null = (Output_unpad_handle == NULL);
  if (!(!main_Output_unpad_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Output_unpad is expected to have non-NULL pointer");
    return -1;
  }
  void* main_Q_unpad_shape = (((DLTensor*)Q_unpad_handle)[0].shape);
  void* main_K_unpad_shape = (((DLTensor*)K_unpad_handle)[0].shape);
  void* main_V_unpad_shape = (((DLTensor*)V_unpad_handle)[0].shape);
  void* main_cu_seqlens_q_shape = (((DLTensor*)cu_seqlens_q_handle)[0].shape);
  void* main_cu_seqlens_k_shape = (((DLTensor*)cu_seqlens_k_handle)[0].shape);
  void* main_Output_unpad_shape = (((DLTensor*)Output_unpad_handle)[0].shape);
  if (!(((((DLTensor*)Q_unpad_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad ndim mismatch, expected 3", (long long)((((DLTensor*)Q_unpad_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Q_unpad_strides = (((DLTensor*)Q_unpad_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)Q_unpad_handle)[0].device.device_id);
  void* Q_unpad = (((DLTensor*)Q_unpad_handle)[0].data);
  if (!(((((DLTensor*)K_unpad_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad ndim mismatch, expected 3", (long long)((((DLTensor*)K_unpad_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_K_unpad_strides = (((DLTensor*)K_unpad_handle)[0].strides);
  void* K_unpad = (((DLTensor*)K_unpad_handle)[0].data);
  if (!(((((DLTensor*)V_unpad_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad ndim mismatch, expected 3", (long long)((((DLTensor*)V_unpad_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_V_unpad_strides = (((DLTensor*)V_unpad_handle)[0].strides);
  void* V_unpad = (((DLTensor*)V_unpad_handle)[0].data);
  if (!(((((DLTensor*)cu_seqlens_q_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_q ndim mismatch, expected 1", (long long)((((DLTensor*)cu_seqlens_q_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_cu_seqlens_q_strides = (((DLTensor*)cu_seqlens_q_handle)[0].strides);
  void* cu_seqlens_q = (((DLTensor*)cu_seqlens_q_handle)[0].data);
  if (!(((((DLTensor*)cu_seqlens_k_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_k ndim mismatch, expected 1", (long long)((((DLTensor*)cu_seqlens_k_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_cu_seqlens_k_strides = (((DLTensor*)cu_seqlens_k_handle)[0].strides);
  void* cu_seqlens_k = (((DLTensor*)cu_seqlens_k_handle)[0].data);
  if (!(((((DLTensor*)Output_unpad_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad ndim mismatch, expected 3", (long long)((((DLTensor*)Output_unpad_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Output_unpad_strides = (((DLTensor*)Output_unpad_handle)[0].strides);
  void* Output_unpad = (((DLTensor*)Output_unpad_handle)[0].data);
  if (!(((((((DLTensor*)Q_unpad_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Q_unpad_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Q_unpad_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q_unpad dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Q_unpad_shape)[0]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Q_unpad_shape)[0])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Q_unpad_shape)[1]) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Q_unpad_shape)[1])), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Q_unpad_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Q_unpad_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_Q_unpad_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)main_Q_unpad_strides)[2]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_Q_unpad_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_Q_unpad_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad strides[2] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_Q_unpad_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_Q_unpad_strides)[1]);
  }
  if (!((condval_2 == 256))) {
    int32_t condval_3;
    if ((main_Q_unpad_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_Q_unpad_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad strides[1] violates packed ABI constraint", (long long)(condval_3), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((main_Q_unpad_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)main_Q_unpad_strides)[0]);
  }
  if (!((condval_4 == 3072))) {
    int32_t condval_5;
    if ((main_Q_unpad_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)main_Q_unpad_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad strides[0] violates packed ABI constraint", (long long)(condval_5), (long long)(3072));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Q_unpad_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Q_unpad_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Q_unpad_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Q_unpad device_type mismatch, expected cuda", (long long)((((DLTensor*)Q_unpad_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Q_unpad == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q_unpad data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)K_unpad_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)K_unpad_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)K_unpad_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_unpad dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_unpad_shape)[0]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_unpad_shape)[0])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_unpad_shape)[1]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_unpad_shape)[1])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_unpad_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_unpad_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((main_K_unpad_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)main_K_unpad_strides)[2]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((main_K_unpad_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)main_K_unpad_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad strides[2] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((main_K_unpad_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)main_K_unpad_strides)[1]);
  }
  if (!((condval_8 == 256))) {
    int32_t condval_9;
    if ((main_K_unpad_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)main_K_unpad_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad strides[1] violates packed ABI constraint", (long long)(condval_9), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((main_K_unpad_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)main_K_unpad_strides)[0]);
  }
  if (!((condval_10 == 512))) {
    int32_t condval_11;
    if ((main_K_unpad_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)main_K_unpad_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad strides[0] violates packed ABI constraint", (long long)(condval_11), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)K_unpad_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)K_unpad_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_unpad_handle)[0].device.device_id) == (((DLTensor*)Q_unpad_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad device_id violates packed ABI constraint", (long long)((((DLTensor*)K_unpad_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_unpad_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_unpad_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_unpad device_type mismatch, expected cuda", (long long)((((DLTensor*)K_unpad_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(K_unpad == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_unpad data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)V_unpad_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)V_unpad_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)V_unpad_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_unpad dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_unpad_shape)[0]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_unpad_shape)[0])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_unpad_shape)[1]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_unpad_shape)[1])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_unpad_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_unpad_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((main_V_unpad_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)main_V_unpad_strides)[2]);
  }
  if (!((condval_12 == 1))) {
    int32_t condval_13;
    if ((main_V_unpad_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)main_V_unpad_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad strides[2] violates packed ABI constraint", (long long)(condval_13), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((main_V_unpad_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)main_V_unpad_strides)[1]);
  }
  if (!((condval_14 == 256))) {
    int32_t condval_15;
    if ((main_V_unpad_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)main_V_unpad_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad strides[1] violates packed ABI constraint", (long long)(condval_15), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((main_V_unpad_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)main_V_unpad_strides)[0]);
  }
  if (!((condval_16 == 512))) {
    int32_t condval_17;
    if ((main_V_unpad_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)main_V_unpad_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad strides[0] violates packed ABI constraint", (long long)(condval_17), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)V_unpad_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)V_unpad_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_unpad_handle)[0].device.device_id) == (((DLTensor*)Q_unpad_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad device_id violates packed ABI constraint", (long long)((((DLTensor*)V_unpad_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_unpad_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_unpad_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_unpad device_type mismatch, expected cuda", (long long)((((DLTensor*)V_unpad_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(V_unpad == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_unpad data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)cu_seqlens_q_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)cu_seqlens_q_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)cu_seqlens_q_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input cu_seqlens_q dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_cu_seqlens_q_shape)[0]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_q shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_cu_seqlens_q_shape)[0])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_18;
  if ((main_cu_seqlens_q_strides == NULL)) {
    condval_18 = 1;
  } else {
    condval_18 = ((int32_t)((int64_t*)main_cu_seqlens_q_strides)[0]);
  }
  if (!((condval_18 == 1))) {
    int32_t condval_19;
    if ((main_cu_seqlens_q_strides == NULL)) {
      condval_19 = 1;
    } else {
      condval_19 = ((int32_t)((int64_t*)main_cu_seqlens_q_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_q strides[0] violates packed ABI constraint", (long long)(condval_19), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)cu_seqlens_q_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_q byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)cu_seqlens_q_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_q_handle)[0].device.device_id) == (((DLTensor*)Q_unpad_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_q device_id violates packed ABI constraint", (long long)((((DLTensor*)cu_seqlens_q_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_unpad_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_q_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_q device_type mismatch, expected cuda", (long long)((((DLTensor*)cu_seqlens_q_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(cu_seqlens_q == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input cu_seqlens_q data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)cu_seqlens_k_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)cu_seqlens_k_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)cu_seqlens_k_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input cu_seqlens_k dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_cu_seqlens_k_shape)[0]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_k shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_cu_seqlens_k_shape)[0])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_20;
  if ((main_cu_seqlens_k_strides == NULL)) {
    condval_20 = 1;
  } else {
    condval_20 = ((int32_t)((int64_t*)main_cu_seqlens_k_strides)[0]);
  }
  if (!((condval_20 == 1))) {
    int32_t condval_21;
    if ((main_cu_seqlens_k_strides == NULL)) {
      condval_21 = 1;
    } else {
      condval_21 = ((int32_t)((int64_t*)main_cu_seqlens_k_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_k strides[0] violates packed ABI constraint", (long long)(condval_21), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)cu_seqlens_k_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_k byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)cu_seqlens_k_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_k_handle)[0].device.device_id) == (((DLTensor*)Q_unpad_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_k device_id violates packed ABI constraint", (long long)((((DLTensor*)cu_seqlens_k_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_unpad_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)cu_seqlens_k_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input cu_seqlens_k device_type mismatch, expected cuda", (long long)((((DLTensor*)cu_seqlens_k_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(cu_seqlens_k == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input cu_seqlens_k data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Output_unpad_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Output_unpad_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Output_unpad_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Output_unpad dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Output_unpad_shape)[0]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Output_unpad_shape)[0])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Output_unpad_shape)[1]) == 12))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Output_unpad_shape)[1])), (long long)(12));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Output_unpad_shape)[2]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Output_unpad_shape)[2])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((main_Output_unpad_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)main_Output_unpad_strides)[2]);
  }
  if (!((condval_22 == 1))) {
    int32_t condval_23;
    if ((main_Output_unpad_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)main_Output_unpad_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad strides[2] violates packed ABI constraint", (long long)(condval_23), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((main_Output_unpad_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)main_Output_unpad_strides)[1]);
  }
  if (!((condval_24 == 256))) {
    int32_t condval_25;
    if ((main_Output_unpad_strides == NULL)) {
      condval_25 = 1;
    } else {
      condval_25 = ((int32_t)((int64_t*)main_Output_unpad_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad strides[1] violates packed ABI constraint", (long long)(condval_25), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((main_Output_unpad_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)main_Output_unpad_strides)[0]);
  }
  if (!((condval_26 == 3072))) {
    int32_t condval_27;
    if ((main_Output_unpad_strides == NULL)) {
      condval_27 = 1;
    } else {
      condval_27 = ((int32_t)((int64_t*)main_Output_unpad_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad strides[0] violates packed ABI constraint", (long long)(condval_27), (long long)(3072));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Output_unpad_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Output_unpad_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_unpad_handle)[0].device.device_id) == (((DLTensor*)Q_unpad_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad device_id violates packed ABI constraint", (long long)((((DLTensor*)Output_unpad_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_unpad_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_unpad_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Output_unpad device_type mismatch, expected cuda", (long long)((((DLTensor*)Output_unpad_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Output_unpad == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Output_unpad data pointer is NULL");
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
  if (K_unpad == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = K_unpad;
  if (Output_unpad == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = Output_unpad;
  if (Q_unpad == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = Q_unpad;
  if (V_unpad == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = V_unpad;
  if (cu_seqlens_k == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = cu_seqlens_k;
  if (cu_seqlens_q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = cu_seqlens_q;
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)max_seqlen_q);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)((max_seqlen_q + 63) >> 6));
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)12);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)163840);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 14, &result_2) != 0) {
    return -1;
  }
  return 0;
}

