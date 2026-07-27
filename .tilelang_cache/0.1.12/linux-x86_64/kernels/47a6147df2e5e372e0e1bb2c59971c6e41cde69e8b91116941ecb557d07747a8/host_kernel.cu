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
  int32_t S_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((S_handle_type_index == 0) || (S_handle_type_index == 4)) || (S_handle_type_index == 7)) || (64 <= S_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input S expected pointer or tensor handle");
    return -1;
  }
  int32_t W_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((W_handle_type_index == 0) || (W_handle_type_index == 4)) || (W_handle_type_index == 7)) || (64 <= W_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W expected pointer or tensor handle");
    return -1;
  }
  int32_t Bias_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((Bias_handle_type_index == 0) || (Bias_handle_type_index == 4)) || (Bias_handle_type_index == 7)) || (64 <= Bias_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias expected pointer or tensor handle");
    return -1;
  }
  int32_t Y_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((Y_handle_type_index == 0) || (Y_handle_type_index == 4)) || (Y_handle_type_index == 7)) || (64 <= Y_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Y expected pointer or tensor handle");
    return -1;
  }
  void* X_handle = ((X_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* S_handle = ((S_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* W_handle = ((W_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* Bias_handle = ((Bias_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* Y_handle = ((Y_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  bool main_X_is_null = (X_handle == NULL);
  if (!(!main_X_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.X is expected to have non-NULL pointer");
    return -1;
  }
  bool main_S_is_null = (S_handle == NULL);
  if (!(!main_S_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.S is expected to have non-NULL pointer");
    return -1;
  }
  bool main_W_is_null = (W_handle == NULL);
  if (!(!main_W_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.W is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Bias_is_null = (Bias_handle == NULL);
  bool main_Y_is_null = (Y_handle == NULL);
  if (!(!main_Y_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Y is expected to have non-NULL pointer");
    return -1;
  }
  void* main_X_shape = (((DLTensor*)X_handle)[0].shape);
  void* main_S_shape = (((DLTensor*)S_handle)[0].shape);
  void* main_W_shape = (((DLTensor*)W_handle)[0].shape);
  void* condval;
  if (!main_Bias_is_null) {
    condval = (((DLTensor*)Bias_handle)[0].shape);
  } else {
    uint64_t v_ = (uint64_t)0;
    condval = (*(void* *)(&(v_)));
  }
  void* main_Bias_shape = condval;
  void* main_Y_shape = (((DLTensor*)Y_handle)[0].shape);
  if (!(((((DLTensor*)X_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X ndim mismatch, expected 2", (long long)((((DLTensor*)X_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_X_strides = (((DLTensor*)X_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)X_handle)[0].device.device_id);
  void* X = (((DLTensor*)X_handle)[0].data);
  if (!(((((DLTensor*)S_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S ndim mismatch, expected 3", (long long)((((DLTensor*)S_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_S_strides = (((DLTensor*)S_handle)[0].strides);
  void* S = (((DLTensor*)S_handle)[0].data);
  if (!(((((DLTensor*)W_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W ndim mismatch, expected 2", (long long)((((DLTensor*)W_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_W_strides = (((DLTensor*)W_handle)[0].strides);
  void* W = (((DLTensor*)W_handle)[0].data);
  if (!((main_Bias_is_null || ((((DLTensor*)Bias_handle)[0].ndim) == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias ndim mismatch, expected 1");
    return -1;
  }
  void* condval_1;
  if (!main_Bias_is_null) {
    condval_1 = (((DLTensor*)Bias_handle)[0].strides);
  } else {
    uint64_t v__1 = (uint64_t)0;
    condval_1 = (*(void* *)(&(v__1)));
  }
  void* main_Bias_strides = condval_1;
  void* condval_2;
  if (!main_Bias_is_null) {
    condval_2 = (((DLTensor*)Bias_handle)[0].data);
  } else {
    uint64_t v__2 = (uint64_t)0;
    condval_2 = (*(void* *)(&(v__2)));
  }
  void* Bias = condval_2;
  if (!(((((DLTensor*)Y_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y ndim mismatch, expected 2", (long long)((((DLTensor*)Y_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Y_strides = (((DLTensor*)Y_handle)[0].strides);
  void* Y = (((DLTensor*)Y_handle)[0].data);
  if (!(((((((DLTensor*)X_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)X_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)X_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input X dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_X_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_X_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_X_shape)[1]) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_X_shape)[1])), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_3;
  if ((main_X_strides == NULL)) {
    condval_3 = 1;
  } else {
    condval_3 = ((int32_t)((int64_t*)main_X_strides)[1]);
  }
  if (!((condval_3 == 1))) {
    int32_t condval_4;
    if ((main_X_strides == NULL)) {
      condval_4 = 1;
    } else {
      condval_4 = ((int32_t)((int64_t*)main_X_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X strides[1] violates packed ABI constraint", (long long)(condval_4), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_5;
  if ((main_X_strides == NULL)) {
    condval_5 = 1;
  } else {
    condval_5 = ((int32_t)((int64_t*)main_X_strides)[0]);
  }
  if (!((condval_5 == 5120))) {
    int32_t condval_6;
    if ((main_X_strides == NULL)) {
      condval_6 = 1;
    } else {
      condval_6 = ((int32_t)((int64_t*)main_X_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input X strides[0] violates packed ABI constraint", (long long)(condval_6), (long long)(5120));
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
  if (!(((((((DLTensor*)S_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)S_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)S_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input S dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_S_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_S_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_S_shape)[1]) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_S_shape)[1])), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_S_shape)[2]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_S_shape)[2])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_7;
  if ((main_S_strides == NULL)) {
    condval_7 = 1;
  } else {
    condval_7 = ((int32_t)((int64_t*)main_S_strides)[2]);
  }
  if (!((condval_7 == 1))) {
    int32_t condval_8;
    if ((main_S_strides == NULL)) {
      condval_8 = 1;
    } else {
      condval_8 = ((int32_t)((int64_t*)main_S_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S strides[2] violates packed ABI constraint", (long long)(condval_8), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_9;
  if ((main_S_strides == NULL)) {
    condval_9 = 1;
  } else {
    condval_9 = ((int32_t)((int64_t*)main_S_strides)[1]);
  }
  if (!((condval_9 == 4))) {
    int32_t condval_10;
    if ((main_S_strides == NULL)) {
      condval_10 = 1;
    } else {
      condval_10 = ((int32_t)((int64_t*)main_S_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S strides[1] violates packed ABI constraint", (long long)(condval_10), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_11;
  if ((main_S_strides == NULL)) {
    condval_11 = 1;
  } else {
    condval_11 = ((int32_t)((int64_t*)main_S_strides)[0]);
  }
  if (!((condval_11 == 20480))) {
    int32_t condval_12;
    if ((main_S_strides == NULL)) {
      condval_12 = 1;
    } else {
      condval_12 = ((int32_t)((int64_t*)main_S_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S strides[0] violates packed ABI constraint", (long long)(condval_12), (long long)(20480));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)S_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)S_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)S_handle)[0].device.device_id) == (((DLTensor*)X_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S device_id violates packed ABI constraint", (long long)((((DLTensor*)S_handle)[0].device.device_id)), (long long)((((DLTensor*)X_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)S_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input S device_type mismatch, expected cuda", (long long)((((DLTensor*)S_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(S == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input S data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)W_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)W_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)W_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input W dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W_shape)[0]) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W_shape)[0])), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_W_shape)[1]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_W_shape)[1])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_13;
  if ((main_W_strides == NULL)) {
    condval_13 = 1;
  } else {
    condval_13 = ((int32_t)((int64_t*)main_W_strides)[1]);
  }
  if (!((condval_13 == 1))) {
    int32_t condval_14;
    if ((main_W_strides == NULL)) {
      condval_14 = 1;
    } else {
      condval_14 = ((int32_t)((int64_t*)main_W_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W strides[1] violates packed ABI constraint", (long long)(condval_14), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_15;
  if ((main_W_strides == NULL)) {
    condval_15 = 1;
  } else {
    condval_15 = ((int32_t)((int64_t*)main_W_strides)[0]);
  }
  if (!((condval_15 == 4))) {
    int32_t condval_16;
    if ((main_W_strides == NULL)) {
      condval_16 = 1;
    } else {
      condval_16 = ((int32_t)((int64_t*)main_W_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input W strides[0] violates packed ABI constraint", (long long)(condval_16), (long long)(4));
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
  uint8_t condval_17;
  if (!main_Bias_is_null) {
    condval_17 = (((DLTensor*)Bias_handle)[0].dtype.code);
  } else {
    condval_17 = (uint8_t)4;
  }
  uint8_t condval_18;
  if (!main_Bias_is_null) {
    condval_18 = (((DLTensor*)Bias_handle)[0].dtype.bits);
  } else {
    condval_18 = (uint8_t)16;
  }
  uint16_t condval_19;
  if (!main_Bias_is_null) {
    condval_19 = (((DLTensor*)Bias_handle)[0].dtype.lanes);
  } else {
    condval_19 = (uint16_t)1;
  }
  if (!((main_Bias_is_null || (((condval_17 == (uint8_t)4) && (condval_18 == (uint8_t)16)) && (condval_19 == (uint16_t)1))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias dtype mismatch, expected bfloat16");
    return -1;
  }
  int32_t condval_20;
  if (!main_Bias_is_null) {
    condval_20 = ((int32_t)((int64_t*)main_Bias_shape)[0]);
  } else {
    condval_20 = 0;
  }
  if (!((main_Bias_is_null || (condval_20 == 5120)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias shape[0] violates packed ABI constraint");
    return -1;
  }
  int32_t condval_21;
  if ((main_Bias_strides == NULL)) {
    condval_21 = 1;
  } else {
    condval_21 = ((int32_t)((int64_t*)main_Bias_strides)[0]);
  }
  if (!((main_Bias_is_null || (condval_21 == 1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias strides[0] violates packed ABI constraint");
    return -1;
  }
  uint64_t condval_22;
  if (!main_Bias_is_null) {
    condval_22 = (((DLTensor*)Bias_handle)[0].byte_offset);
  } else {
    condval_22 = (uint64_t)0;
  }
  if (!((main_Bias_is_null || ((uint64_t)0 == condval_22)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias byte_offset violates packed ABI constraint");
    return -1;
  }
  int32_t condval_23;
  if (!main_Bias_is_null) {
    condval_23 = (((DLTensor*)Bias_handle)[0].device.device_id);
  } else {
    condval_23 = 0;
  }
  if (!((main_Bias_is_null || (condval_23 == (((DLTensor*)X_handle)[0].device.device_id))))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias device_id violates packed ABI constraint");
    return -1;
  }
  int32_t condval_24;
  if (!main_Bias_is_null) {
    condval_24 = (((DLTensor*)Bias_handle)[0].device.device_type);
  } else {
    condval_24 = 0;
  }
  if (!((main_Bias_is_null || (condval_24 == 2)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias device_type mismatch, expected cuda");
    return -1;
  }
  if (!((main_Bias_is_null || !(Bias == NULL)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Bias data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Y_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Y_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Y_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Y dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Y_shape)[0]) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Y_shape)[0])), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Y_shape)[1]) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Y_shape)[1])), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_25;
  if ((main_Y_strides == NULL)) {
    condval_25 = 1;
  } else {
    condval_25 = ((int32_t)((int64_t*)main_Y_strides)[1]);
  }
  if (!((condval_25 == 1))) {
    int32_t condval_26;
    if ((main_Y_strides == NULL)) {
      condval_26 = 1;
    } else {
      condval_26 = ((int32_t)((int64_t*)main_Y_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y strides[1] violates packed ABI constraint", (long long)(condval_26), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_27;
  if ((main_Y_strides == NULL)) {
    condval_27 = 1;
  } else {
    condval_27 = ((int32_t)((int64_t*)main_Y_strides)[0]);
  }
  if (!((condval_27 == 5120))) {
    int32_t condval_28;
    if ((main_Y_strides == NULL)) {
      condval_28 = 1;
    } else {
      condval_28 = ((int32_t)((int64_t*)main_Y_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Y strides[0] violates packed ABI constraint", (long long)(condval_28), (long long)(5120));
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
  if (S == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = S;
  if (W == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = W;
  if (X == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = X;
  if (Y == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = Y;
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)40);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
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

