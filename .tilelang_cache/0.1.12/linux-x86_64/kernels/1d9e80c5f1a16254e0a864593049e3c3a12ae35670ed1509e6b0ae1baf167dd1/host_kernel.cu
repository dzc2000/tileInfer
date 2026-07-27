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
  TL_ALIGN(128) TVMFFIAny stack[12];
  void* stack_ffi_any = stack;
  if (!((num_args == 5))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 5", (long long)(num_args), (long long)(5));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t X_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((X_handle_type_index == 0) || (X_handle_type_index == 4)) || (X_handle_type_index == 7)) || (64 <= X_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input X expected pointer or tensor handle");
    return -1;
  }
  int32_t W_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((W_handle_type_index == 0) || (W_handle_type_index == 4)) || (W_handle_type_index == 7)) || (64 <= W_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W expected pointer or tensor handle");
    return -1;
  }
  int32_t COS_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((COS_handle_type_index == 0) || (COS_handle_type_index == 4)) || (COS_handle_type_index == 7)) || (64 <= COS_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input COS expected pointer or tensor handle");
    return -1;
  }
  int32_t SIN_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((SIN_handle_type_index == 0) || (SIN_handle_type_index == 4)) || (SIN_handle_type_index == 7)) || (64 <= SIN_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SIN expected pointer or tensor handle");
    return -1;
  }
  int32_t Y_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((Y_handle_type_index == 0) || (Y_handle_type_index == 4)) || (Y_handle_type_index == 7)) || (64 <= Y_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Y expected pointer or tensor handle");
    return -1;
  }
  void* X_handle = ((X_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* W_handle = ((W_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* COS_handle = ((COS_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* SIN_handle = ((SIN_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* Y_handle = ((Y_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  bool main_X_is_null = (X_handle == NULL);
  if (!(!main_X_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.X is expected to have non-NULL pointer");
    return -1;
  }
  bool main_W_is_null = (W_handle == NULL);
  if (!(!main_W_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.W is expected to have non-NULL pointer");
    return -1;
  }
  bool main_COS_is_null = (COS_handle == NULL);
  if (!(!main_COS_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.COS is expected to have non-NULL pointer");
    return -1;
  }
  bool main_SIN_is_null = (SIN_handle == NULL);
  if (!(!main_SIN_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.SIN is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Y_is_null = (Y_handle == NULL);
  if (!(!main_Y_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Y is expected to have non-NULL pointer");
    return -1;
  }
  void* main_X_shape = (((DLTensor*)X_handle)[0].shape);
  void* main_W_shape = (((DLTensor*)W_handle)[0].shape);
  void* main_COS_shape = (((DLTensor*)COS_handle)[0].shape);
  void* main_SIN_shape = (((DLTensor*)SIN_handle)[0].shape);
  void* main_Y_shape = (((DLTensor*)Y_handle)[0].shape);
  if (!(((((DLTensor*)X_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X ndim mismatch, expected 4", (long long)((((DLTensor*)X_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_X_strides = (((DLTensor*)X_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)X_handle)[0].device.device_id);
  void* X = (((DLTensor*)X_handle)[0].data);
  if (!(((((DLTensor*)W_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W ndim mismatch, expected 1", (long long)((((DLTensor*)W_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_W_strides = (((DLTensor*)W_handle)[0].strides);
  void* W = (((DLTensor*)W_handle)[0].data);
  if (!(((((DLTensor*)COS_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input COS ndim mismatch, expected 1", (long long)((((DLTensor*)COS_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_COS_strides = (((DLTensor*)COS_handle)[0].strides);
  void* COS = (((DLTensor*)COS_handle)[0].data);
  if (!(((((DLTensor*)SIN_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SIN ndim mismatch, expected 1", (long long)((((DLTensor*)SIN_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_SIN_strides = (((DLTensor*)SIN_handle)[0].strides);
  void* SIN = (((DLTensor*)SIN_handle)[0].data);
  if (!(((((DLTensor*)Y_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y ndim mismatch, expected 4", (long long)((((DLTensor*)Y_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Y_strides = (((DLTensor*)Y_handle)[0].strides);
  void* Y = (((DLTensor*)Y_handle)[0].data);
  if (!(((((((DLTensor*)X_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)X_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)X_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input X dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_X_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_X_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_X_shape)[1]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_X_shape)[1])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_X_shape)[2]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_X_shape)[2])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_X_shape)[3]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_X_shape)[3])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_X_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)main_X_strides)[3]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_X_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_X_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X strides[3] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_X_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_X_strides)[1]);
  }
  if (!((condval_2 == 256))) {
    int32_t condval_3;
    if ((main_X_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_X_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X strides[1] violates packed ABI constraint", (long long)(condval_3), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)X_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)X_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)X_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X device_type mismatch, expected cuda", (long long)((((DLTensor*)X_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(X == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input X data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)W_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)W_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)W_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W_shape)[0]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W_shape)[0])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((main_W_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)main_W_strides)[0]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((main_W_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)main_W_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W strides[0] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)W_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)W_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)W_handle)[0].device.device_id) == (((DLTensor*)X_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W device_id violates packed ABI constraint", (long long)((((DLTensor*)W_handle)[0].device.device_id)), (long long)((((DLTensor*)X_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)W_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W device_type mismatch, expected cuda", (long long)((((DLTensor*)W_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(W == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)COS_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)COS_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)COS_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input COS dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_COS_shape)[0]) == 64))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input COS shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_COS_shape)[0])), (long long)(64));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((main_COS_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)main_COS_strides)[0]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((main_COS_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)main_COS_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input COS strides[0] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)COS_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input COS byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)COS_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)COS_handle)[0].device.device_id) == (((DLTensor*)X_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input COS device_id violates packed ABI constraint", (long long)((((DLTensor*)COS_handle)[0].device.device_id)), (long long)((((DLTensor*)X_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)COS_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input COS device_type mismatch, expected cuda", (long long)((((DLTensor*)COS_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(COS == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input COS data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)SIN_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)SIN_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)SIN_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SIN dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_SIN_shape)[0]) == 64))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SIN shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_SIN_shape)[0])), (long long)(64));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((main_SIN_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)main_SIN_strides)[0]);
  }
  if (!((condval_8 == 1))) {
    int32_t condval_9;
    if ((main_SIN_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)main_SIN_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SIN strides[0] violates packed ABI constraint", (long long)(condval_9), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)SIN_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SIN byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)SIN_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SIN_handle)[0].device.device_id) == (((DLTensor*)X_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SIN device_id violates packed ABI constraint", (long long)((((DLTensor*)SIN_handle)[0].device.device_id)), (long long)((((DLTensor*)X_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)SIN_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input SIN device_type mismatch, expected cuda", (long long)((((DLTensor*)SIN_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(SIN == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input SIN data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Y_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Y_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Y_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Y dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Y_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Y_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Y_shape)[1]) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Y_shape)[1])), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Y_shape)[2]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Y_shape)[2])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Y_shape)[3]) == 256))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Y_shape)[3])), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((main_Y_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)main_Y_strides)[3]);
  }
  if (!((condval_10 == 1))) {
    int32_t condval_11;
    if ((main_Y_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)main_Y_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y strides[3] violates packed ABI constraint", (long long)(condval_11), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((main_Y_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)main_Y_strides)[1]);
  }
  if (!((condval_12 == 256))) {
    int32_t condval_13;
    if ((main_Y_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)main_Y_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y strides[1] violates packed ABI constraint", (long long)(condval_13), (long long)(256));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Y_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Y_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Y_handle)[0].device.device_id) == (((DLTensor*)X_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y device_id violates packed ABI constraint", (long long)((((DLTensor*)Y_handle)[0].device.device_id)), (long long)((((DLTensor*)X_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Y_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y device_type mismatch, expected cuda", (long long)((((DLTensor*)Y_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Y == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Y data pointer is NULL");
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
  if (COS == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = COS;
  if (SIN == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = SIN;
  if (W == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = W;
  if (X == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = X;
  if (Y == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = Y;
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)1664);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 11, &result_2) != 0) {
    return -1;
  }
  return 0;
}

