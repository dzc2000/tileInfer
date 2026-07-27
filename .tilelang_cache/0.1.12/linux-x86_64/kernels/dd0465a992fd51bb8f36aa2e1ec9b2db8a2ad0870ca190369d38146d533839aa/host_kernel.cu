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
  TL_ALIGN(128) TVMFFIAny stack[11];
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
  int32_t Key_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((Key_handle_type_index == 0) || (Key_handle_type_index == 4)) || (Key_handle_type_index == 7)) || (64 <= Key_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Key expected pointer or tensor handle");
    return -1;
  }
  int32_t Value_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((Value_handle_type_index == 0) || (Value_handle_type_index == 4)) || (Value_handle_type_index == 7)) || (64 <= Value_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Value expected pointer or tensor handle");
    return -1;
  }
  int32_t Slot_Mapping_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((Slot_Mapping_handle_type_index == 0) || (Slot_Mapping_handle_type_index == 4)) || (Slot_Mapping_handle_type_index == 7)) || (64 <= Slot_Mapping_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Slot_Mapping expected pointer or tensor handle");
    return -1;
  }
  int32_t K_Cache_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((K_Cache_handle_type_index == 0) || (K_Cache_handle_type_index == 4)) || (K_Cache_handle_type_index == 7)) || (64 <= K_Cache_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_Cache expected pointer or tensor handle");
    return -1;
  }
  int32_t V_Cache_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((V_Cache_handle_type_index == 0) || (V_Cache_handle_type_index == 4)) || (V_Cache_handle_type_index == 7)) || (64 <= V_Cache_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_Cache expected pointer or tensor handle");
    return -1;
  }
  void* Key_handle = ((Key_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* Value_handle = ((Value_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* Slot_Mapping_handle = ((Slot_Mapping_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* K_Cache_handle = ((K_Cache_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* V_Cache_handle = ((V_Cache_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  bool main_Key_is_null = (Key_handle == NULL);
  if (!(!main_Key_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Key is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Value_is_null = (Value_handle == NULL);
  if (!(!main_Value_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Value is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Slot_Mapping_is_null = (Slot_Mapping_handle == NULL);
  if (!(!main_Slot_Mapping_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Slot_Mapping is expected to have non-NULL pointer");
    return -1;
  }
  bool main_K_Cache_is_null = (K_Cache_handle == NULL);
  if (!(!main_K_Cache_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.K_Cache is expected to have non-NULL pointer");
    return -1;
  }
  bool main_V_Cache_is_null = (V_Cache_handle == NULL);
  if (!(!main_V_Cache_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.V_Cache is expected to have non-NULL pointer");
    return -1;
  }
  void* main_Key_shape = (((DLTensor*)Key_handle)[0].shape);
  void* main_Value_shape = (((DLTensor*)Value_handle)[0].shape);
  void* main_Slot_Mapping_shape = (((DLTensor*)Slot_Mapping_handle)[0].shape);
  void* main_K_Cache_shape = (((DLTensor*)K_Cache_handle)[0].shape);
  void* main_V_Cache_shape = (((DLTensor*)V_Cache_handle)[0].shape);
  if (!(((((DLTensor*)Key_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key ndim mismatch, expected 2", (long long)((((DLTensor*)Key_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Key_strides = (((DLTensor*)Key_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)Key_handle)[0].device.device_id);
  void* Key = (((DLTensor*)Key_handle)[0].data);
  if (!(((((DLTensor*)Value_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value ndim mismatch, expected 2", (long long)((((DLTensor*)Value_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Value_strides = (((DLTensor*)Value_handle)[0].strides);
  void* Value = (((DLTensor*)Value_handle)[0].data);
  if (!(((((DLTensor*)Slot_Mapping_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Slot_Mapping ndim mismatch, expected 1", (long long)((((DLTensor*)Slot_Mapping_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_Slot_Mapping_strides = (((DLTensor*)Slot_Mapping_handle)[0].strides);
  void* Slot_Mapping = (((DLTensor*)Slot_Mapping_handle)[0].data);
  if (!(((((DLTensor*)K_Cache_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache ndim mismatch, expected 2", (long long)((((DLTensor*)K_Cache_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_K_Cache_strides = (((DLTensor*)K_Cache_handle)[0].strides);
  void* K_Cache = (((DLTensor*)K_Cache_handle)[0].data);
  if (!(((((DLTensor*)V_Cache_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache ndim mismatch, expected 2", (long long)((((DLTensor*)V_Cache_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* main_V_Cache_strides = (((DLTensor*)V_Cache_handle)[0].strides);
  void* V_Cache = (((DLTensor*)V_Cache_handle)[0].data);
  if (!(((((((DLTensor*)Key_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Key_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Key_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Key dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Key_shape)[0]) == 5))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Key_shape)[0])), (long long)(5));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Key_shape)[1]) == 512))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Key_shape)[1])), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((main_Key_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)main_Key_strides)[1]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((main_Key_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)main_Key_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key strides[1] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((main_Key_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)main_Key_strides)[0]);
  }
  if (!((condval_2 == 512))) {
    int32_t condval_3;
    if ((main_Key_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)main_Key_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key strides[0] violates packed ABI constraint", (long long)(condval_3), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Key_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Key_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Key_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Key device_type mismatch, expected cuda", (long long)((((DLTensor*)Key_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Key == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Key data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Value_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Value_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Value_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Value dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Value_shape)[0]) == 5))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Value_shape)[0])), (long long)(5));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Value_shape)[1]) == 512))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Value_shape)[1])), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((main_Value_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)main_Value_strides)[1]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((main_Value_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)main_Value_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value strides[1] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((main_Value_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)main_Value_strides)[0]);
  }
  if (!((condval_6 == 512))) {
    int32_t condval_7;
    if ((main_Value_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)main_Value_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value strides[0] violates packed ABI constraint", (long long)(condval_7), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Value_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Value_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Value_handle)[0].device.device_id) == (((DLTensor*)Key_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value device_id violates packed ABI constraint", (long long)((((DLTensor*)Value_handle)[0].device.device_id)), (long long)((((DLTensor*)Key_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Value_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Value device_type mismatch, expected cuda", (long long)((((DLTensor*)Value_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Value == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Value data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Slot_Mapping_handle)[0].dtype.code) == (uint8_t)0) && ((((DLTensor*)Slot_Mapping_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)Slot_Mapping_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Slot_Mapping dtype mismatch, expected int32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_Slot_Mapping_shape)[0]) == 5))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Slot_Mapping shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_Slot_Mapping_shape)[0])), (long long)(5));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((main_Slot_Mapping_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)main_Slot_Mapping_strides)[0]);
  }
  if (!((condval_8 == 1))) {
    int32_t condval_9;
    if ((main_Slot_Mapping_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)main_Slot_Mapping_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Slot_Mapping strides[0] violates packed ABI constraint", (long long)(condval_9), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Slot_Mapping_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Slot_Mapping byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Slot_Mapping_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Slot_Mapping_handle)[0].device.device_id) == (((DLTensor*)Key_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Slot_Mapping device_id violates packed ABI constraint", (long long)((((DLTensor*)Slot_Mapping_handle)[0].device.device_id)), (long long)((((DLTensor*)Key_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Slot_Mapping_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input Slot_Mapping device_type mismatch, expected cuda", (long long)((((DLTensor*)Slot_Mapping_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Slot_Mapping == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Slot_Mapping data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)K_Cache_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)K_Cache_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)K_Cache_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_Cache dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_Cache_shape)[0]) == 293536))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_Cache_shape)[0])), (long long)(293536));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_K_Cache_shape)[1]) == 512))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_K_Cache_shape)[1])), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((main_K_Cache_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)main_K_Cache_strides)[1]);
  }
  if (!((condval_10 == 1))) {
    int32_t condval_11;
    if ((main_K_Cache_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)main_K_Cache_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache strides[1] violates packed ABI constraint", (long long)(condval_11), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((main_K_Cache_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)main_K_Cache_strides)[0]);
  }
  if (!((condval_12 == 512))) {
    int32_t condval_13;
    if ((main_K_Cache_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)main_K_Cache_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache strides[0] violates packed ABI constraint", (long long)(condval_13), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)K_Cache_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)K_Cache_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_Cache_handle)[0].device.device_id) == (((DLTensor*)Key_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache device_id violates packed ABI constraint", (long long)((((DLTensor*)K_Cache_handle)[0].device.device_id)), (long long)((((DLTensor*)Key_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_Cache_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input K_Cache device_type mismatch, expected cuda", (long long)((((DLTensor*)K_Cache_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(K_Cache == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K_Cache data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)V_Cache_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)V_Cache_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)V_Cache_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_Cache dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_Cache_shape)[0]) == 293536))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_Cache_shape)[0])), (long long)(293536));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)main_V_Cache_shape)[1]) == 512))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)main_V_Cache_shape)[1])), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((main_V_Cache_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)main_V_Cache_strides)[1]);
  }
  if (!((condval_14 == 1))) {
    int32_t condval_15;
    if ((main_V_Cache_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)main_V_Cache_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache strides[1] violates packed ABI constraint", (long long)(condval_15), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((main_V_Cache_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)main_V_Cache_strides)[0]);
  }
  if (!((condval_16 == 512))) {
    int32_t condval_17;
    if ((main_V_Cache_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)main_V_Cache_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache strides[0] violates packed ABI constraint", (long long)(condval_17), (long long)(512));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)V_Cache_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)V_Cache_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_Cache_handle)[0].device.device_id) == (((DLTensor*)Key_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache device_id violates packed ABI constraint", (long long)((((DLTensor*)V_Cache_handle)[0].device.device_id)), (long long)((((DLTensor*)Key_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_Cache_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel main input V_Cache device_type mismatch, expected cuda", (long long)((((DLTensor*)V_Cache_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(V_Cache == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V_Cache data pointer is NULL");
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
  if (K_Cache == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = K_Cache;
  if (Key == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = Key;
  if (Slot_Mapping == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = Slot_Mapping;
  if (V_Cache == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = V_Cache;
  if (Value == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = Value;
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)32832);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 10, &result_2) != 0) {
    return -1;
  }
  return 0;
}

