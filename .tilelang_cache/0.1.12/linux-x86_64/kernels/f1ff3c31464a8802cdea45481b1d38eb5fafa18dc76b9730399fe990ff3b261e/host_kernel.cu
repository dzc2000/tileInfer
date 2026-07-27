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
  TL_ALIGN(128) TVMFFIAny stack[8];
  void* stack_ffi_any = stack;
  if (!((num_args == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 3", (long long)(num_args), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t A_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((A_handle_type_index == 0) || (A_handle_type_index == 4)) || (A_handle_type_index == 7)) || (64 <= A_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A expected pointer or tensor handle");
    return -1;
  }
  int32_t B_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((B_handle_type_index == 0) || (B_handle_type_index == 4)) || (B_handle_type_index == 7)) || (64 <= B_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input B expected pointer or tensor handle");
    return -1;
  }
  int32_t C_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((C_handle_type_index == 0) || (C_handle_type_index == 4)) || (C_handle_type_index == 7)) || (64 <= C_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C expected pointer or tensor handle");
    return -1;
  }
  void* A_handle = ((A_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* B_handle = ((B_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* C_handle = ((C_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  bool main_A_is_null = (A_handle == NULL);
  if (!(!main_A_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.A is expected to have non-NULL pointer");
    return -1;
  }
  bool main_B_is_null = (B_handle == NULL);
  if (!(!main_B_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.B is expected to have non-NULL pointer");
    return -1;
  }
  bool main_C_is_null = (C_handle == NULL);
  if (!(!main_C_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.C is expected to have non-NULL pointer");
    return -1;
  }
  void* main_A_shape = (((DLTensor*)A_handle)[0].shape);
  void* main_B_shape = (((DLTensor*)B_handle)[0].shape);
  void* main_C_shape = (((DLTensor*)C_handle)[0].shape);
  if (!(((((DLTensor*)A_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A ndim mismatch, expected 1", (long long)((((DLTensor*)A_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_A_strides = (((DLTensor*)A_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)A_handle)[0].device.device_id);
  void* A = (((DLTensor*)A_handle)[0].data);
  if (!(((((DLTensor*)B_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B ndim mismatch, expected 1", (long long)((((DLTensor*)B_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_B_strides = (((DLTensor*)B_handle)[0].strides);
  void* B = (((DLTensor*)B_handle)[0].data);
  if (!(((((DLTensor*)C_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C ndim mismatch, expected 1", (long long)((((DLTensor*)C_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_C_strides = (((DLTensor*)C_handle)[0].strides);
  void* C = (((DLTensor*)C_handle)[0].data);
  if (!(((((((DLTensor*)A_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)A_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)A_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_A_shape)[0]) == 139264))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_A_shape)[0])), (long long)(139264));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_A_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)main_A_strides)[0]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_A_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_A_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A strides[0] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)A_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)A_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)A_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input A device_type mismatch, expected cuda", (long long)((((DLTensor*)A_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(A == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input A data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)B_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)B_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)B_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input B dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_B_shape)[0]) == 139264))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_B_shape)[0])), (long long)(139264));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_B_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_B_strides)[0]);
  }
  if (!((condval_2 == 1))) {
    int32_t condval_3;
    if ((main_B_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_B_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B strides[0] violates packed ABI constraint", (long long)(condval_3), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)B_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)B_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)B_handle)[0].device.device_id) == (((DLTensor*)A_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B device_id violates packed ABI constraint", (long long)((((DLTensor*)B_handle)[0].device.device_id)), (long long)((((DLTensor*)A_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)B_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input B device_type mismatch, expected cuda", (long long)((((DLTensor*)B_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(B == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input B data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)C_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)C_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)C_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_C_shape)[0]) == 139264))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_C_shape)[0])), (long long)(139264));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((main_C_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)main_C_strides)[0]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((main_C_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)main_C_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C strides[0] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)C_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)C_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)C_handle)[0].device.device_id) == (((DLTensor*)A_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C device_id violates packed ABI constraint", (long long)((((DLTensor*)C_handle)[0].device.device_id)), (long long)((((DLTensor*)A_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)C_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input C device_type mismatch, expected cuda", (long long)((((DLTensor*)C_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(C == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input C data pointer is NULL");
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
  if (A == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A;
  if (B == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = B;
  if (C == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = C;
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)68);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 7, &result_2) != 0) {
    return -1;
  }
  return 0;
}

