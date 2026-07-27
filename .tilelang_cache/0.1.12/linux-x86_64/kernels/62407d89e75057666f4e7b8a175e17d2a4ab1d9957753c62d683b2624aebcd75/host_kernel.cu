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
  TL_ALIGN(128) TVMFFIAny stack[10];
  void* stack_ffi_any = stack;
  if (!((num_args == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 4", (long long)(num_args), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t partial_max_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((partial_max_handle_type_index == 0) || (partial_max_handle_type_index == 4)) || (partial_max_handle_type_index == 7)) || (64 <= partial_max_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input partial_max expected pointer or tensor handle");
    return -1;
  }
  int32_t partial_idx_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((partial_idx_handle_type_index == 0) || (partial_idx_handle_type_index == 4)) || (partial_idx_handle_type_index == 7)) || (64 <= partial_idx_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input partial_idx expected pointer or tensor handle");
    return -1;
  }
  int32_t out_idx_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((out_idx_handle_type_index == 0) || (out_idx_handle_type_index == 4)) || (out_idx_handle_type_index == 7)) || (64 <= out_idx_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input out_idx expected pointer or tensor handle");
    return -1;
  }
  int32_t out_max_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((out_max_handle_type_index == 0) || (out_max_handle_type_index == 4)) || (out_max_handle_type_index == 7)) || (64 <= out_max_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input out_max expected pointer or tensor handle");
    return -1;
  }
  void* partial_max_handle = ((partial_max_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* partial_idx_handle = ((partial_idx_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* out_idx_handle = ((out_idx_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* out_max_handle = ((out_max_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  bool main_partial_max_is_null = (partial_max_handle == NULL);
  if (!(!main_partial_max_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.partial_max is expected to have non-NULL pointer");
    return -1;
  }
  bool main_partial_idx_is_null = (partial_idx_handle == NULL);
  if (!(!main_partial_idx_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.partial_idx is expected to have non-NULL pointer");
    return -1;
  }
  bool main_out_idx_is_null = (out_idx_handle == NULL);
  if (!(!main_out_idx_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.out_idx is expected to have non-NULL pointer");
    return -1;
  }
  bool main_out_max_is_null = (out_max_handle == NULL);
  if (!(!main_out_max_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.out_max is expected to have non-NULL pointer");
    return -1;
  }
  void* main_partial_max_shape = (((DLTensor*)partial_max_handle)[0].shape);
  void* main_partial_idx_shape = (((DLTensor*)partial_idx_handle)[0].shape);
  void* main_out_idx_shape = (((DLTensor*)out_idx_handle)[0].shape);
  void* main_out_max_shape = (((DLTensor*)out_max_handle)[0].shape);
  if (!(((((DLTensor*)partial_max_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_max ndim mismatch, expected 2", (long long)((((DLTensor*)partial_max_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_partial_max_strides = (((DLTensor*)partial_max_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)partial_max_handle)[0].device.device_id);
  void* partial_max = (((DLTensor*)partial_max_handle)[0].data);
  if (!(((((DLTensor*)partial_idx_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx ndim mismatch, expected 2", (long long)((((DLTensor*)partial_idx_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_partial_idx_strides = (((DLTensor*)partial_idx_handle)[0].strides);
  void* partial_idx = (((DLTensor*)partial_idx_handle)[0].data);
  if (!(((((DLTensor*)out_idx_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_idx ndim mismatch, expected 1", (long long)((((DLTensor*)out_idx_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_out_idx_strides = (((DLTensor*)out_idx_handle)[0].strides);
  void* out_idx = (((DLTensor*)out_idx_handle)[0].data);
  if (!(((((DLTensor*)out_max_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_max ndim mismatch, expected 1", (long long)((((DLTensor*)out_max_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_out_max_strides = (((DLTensor*)out_max_handle)[0].strides);
  void* out_max = (((DLTensor*)out_max_handle)[0].data);
  if (!(((((((DLTensor*)partial_max_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)partial_max_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)partial_max_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input partial_max dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_partial_max_shape)[0]) == 970))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_max shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_partial_max_shape)[0])), (long long)(970));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_partial_max_shape)[1]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_max shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_partial_max_shape)[1])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_partial_max_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)main_partial_max_strides)[0]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_partial_max_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_partial_max_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_max strides[0] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)partial_max_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_max byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)partial_max_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)partial_max_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_max device_type mismatch, expected cuda", (long long)((((DLTensor*)partial_max_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(partial_max == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input partial_max data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)partial_idx_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)partial_idx_handle)[0].dtype.bits) == (uint8_t)64)) && ((((DLTensor*)partial_idx_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input partial_idx dtype mismatch, expected int64");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_partial_idx_shape)[0]) == 970))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_partial_idx_shape)[0])), (long long)(970));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_partial_idx_shape)[1]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_partial_idx_shape)[1])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_partial_idx_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_partial_idx_strides)[0]);
  }
  if (!((condval_2 == 1))) {
    int32_t condval_3;
    if ((main_partial_idx_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_partial_idx_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx strides[0] violates packed ABI constraint", (long long)(condval_3), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)partial_idx_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)partial_idx_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)partial_idx_handle)[0].device.device_id) == (((DLTensor*)partial_max_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx device_id violates packed ABI constraint", (long long)((((DLTensor*)partial_idx_handle)[0].device.device_id)), (long long)((((DLTensor*)partial_max_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)partial_idx_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input partial_idx device_type mismatch, expected cuda", (long long)((((DLTensor*)partial_idx_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(partial_idx == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input partial_idx data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)out_idx_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)out_idx_handle)[0].dtype.bits) == (uint8_t)64)) && ((((DLTensor*)out_idx_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input out_idx dtype mismatch, expected int64");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_out_idx_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_idx shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_out_idx_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)out_idx_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_idx byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)out_idx_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_idx_handle)[0].device.device_id) == (((DLTensor*)partial_max_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_idx device_id violates packed ABI constraint", (long long)((((DLTensor*)out_idx_handle)[0].device.device_id)), (long long)((((DLTensor*)partial_max_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_idx_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_idx device_type mismatch, expected cuda", (long long)((((DLTensor*)out_idx_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(out_idx == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input out_idx data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)out_max_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)out_max_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)out_max_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input out_max dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_out_max_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_max shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_out_max_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)out_max_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_max byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)out_max_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_max_handle)[0].device.device_id) == (((DLTensor*)partial_max_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_max device_id violates packed ABI constraint", (long long)((((DLTensor*)out_max_handle)[0].device.device_id)), (long long)((((DLTensor*)partial_max_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)out_max_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input out_max device_type mismatch, expected cuda", (long long)((((DLTensor*)out_max_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(out_max == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input out_max data pointer is NULL");
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
  if (out_idx == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = out_idx;
  if (out_max == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = out_max;
  if (partial_idx == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = partial_idx;
  if (partial_max == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = partial_max;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)24576);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 9, &result_2) != 0) {
    return -1;
  }
  return 0;
}

