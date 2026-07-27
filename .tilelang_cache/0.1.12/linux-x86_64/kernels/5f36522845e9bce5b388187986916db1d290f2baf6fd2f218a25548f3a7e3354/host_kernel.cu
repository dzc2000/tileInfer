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
static void* kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TL_ALIGN(128) TVMFFIAny stack[17];
  void* stack_ffi_any = stack;
  if (!((num_args == 10))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel: num_args should be 10", (long long)(num_args), (long long)(10));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel: args pointer is NULL");
    return -1;
  }
  int32_t QKV_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((QKV_handle_type_index == 0) || (QKV_handle_type_index == 4)) || (QKV_handle_type_index == 7)) || (64 <= QKV_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input QKV expected pointer or tensor handle");
    return -1;
  }
  int32_t Alpha_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((Alpha_handle_type_index == 0) || (Alpha_handle_type_index == 4)) || (Alpha_handle_type_index == 7)) || (64 <= Alpha_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Alpha expected pointer or tensor handle");
    return -1;
  }
  int32_t BetaRaw_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((BetaRaw_handle_type_index == 0) || (BetaRaw_handle_type_index == 4)) || (BetaRaw_handle_type_index == 7)) || (64 <= BetaRaw_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input BetaRaw expected pointer or tensor handle");
    return -1;
  }
  int32_t NegAExp_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((NegAExp_handle_type_index == 0) || (NegAExp_handle_type_index == 4)) || (NegAExp_handle_type_index == 7)) || (64 <= NegAExp_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NegAExp expected pointer or tensor handle");
    return -1;
  }
  int32_t DtBias_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((DtBias_handle_type_index == 0) || (DtBias_handle_type_index == 4)) || (DtBias_handle_type_index == 7)) || (64 <= DtBias_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input DtBias expected pointer or tensor handle");
    return -1;
  }
  int32_t State_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((State_handle_type_index == 0) || (State_handle_type_index == 4)) || (State_handle_type_index == 7)) || (64 <= State_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input State expected pointer or tensor handle");
    return -1;
  }
  int32_t NormW_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((NormW_handle_type_index == 0) || (NormW_handle_type_index == 4)) || (NormW_handle_type_index == 7)) || (64 <= NormW_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NormW expected pointer or tensor handle");
    return -1;
  }
  int32_t Z_handle_type_index = (((TVMFFIAny*)args)[7].type_index);
  if (!(((((Z_handle_type_index == 0) || (Z_handle_type_index == 4)) || (Z_handle_type_index == 7)) || (64 <= Z_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Z expected pointer or tensor handle");
    return -1;
  }
  int32_t NewState_handle_type_index = (((TVMFFIAny*)args)[8].type_index);
  if (!(((((NewState_handle_type_index == 0) || (NewState_handle_type_index == 4)) || (NewState_handle_type_index == 7)) || (64 <= NewState_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NewState expected pointer or tensor handle");
    return -1;
  }
  int32_t Output_handle_type_index = (((TVMFFIAny*)args)[9].type_index);
  if (!(((((Output_handle_type_index == 0) || (Output_handle_type_index == 4)) || (Output_handle_type_index == 7)) || (64 <= Output_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Output expected pointer or tensor handle");
    return -1;
  }
  void* QKV_handle = ((QKV_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* Alpha_handle = ((Alpha_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* BetaRaw_handle = ((BetaRaw_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* NegAExp_handle = ((NegAExp_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* DtBias_handle = ((DtBias_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* State_handle = ((State_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  void* NormW_handle = ((NormW_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  void* Z_handle = ((Z_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + 24)) : (((TVMFFIAny*)args)[7].v_ptr));
  void* NewState_handle = ((NewState_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[8].v_ptr) + 24)) : (((TVMFFIAny*)args)[8].v_ptr));
  void* Output_handle = ((Output_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[9].v_ptr) + 24)) : (((TVMFFIAny*)args)[9].v_ptr));
  bool kernel_QKV_is_null = (QKV_handle == NULL);
  if (!(!kernel_QKV_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.QKV is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_Alpha_is_null = (Alpha_handle == NULL);
  if (!(!kernel_Alpha_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.Alpha is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_BetaRaw_is_null = (BetaRaw_handle == NULL);
  if (!(!kernel_BetaRaw_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.BetaRaw is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_NegAExp_is_null = (NegAExp_handle == NULL);
  if (!(!kernel_NegAExp_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.NegAExp is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_DtBias_is_null = (DtBias_handle == NULL);
  if (!(!kernel_DtBias_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.DtBias is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_State_is_null = (State_handle == NULL);
  if (!(!kernel_State_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.State is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_NormW_is_null = (NormW_handle == NULL);
  if (!(!kernel_NormW_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.NormW is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_Z_is_null = (Z_handle == NULL);
  if (!(!kernel_Z_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.Z is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_NewState_is_null = (NewState_handle == NULL);
  if (!(!kernel_NewState_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.NewState is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_Output_is_null = (Output_handle == NULL);
  if (!(!kernel_Output_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.Output is expected to have non-NULL pointer");
    return -1;
  }
  void* kernel_QKV_shape = (((DLTensor*)QKV_handle)[0].shape);
  void* kernel_Alpha_shape = (((DLTensor*)Alpha_handle)[0].shape);
  void* kernel_BetaRaw_shape = (((DLTensor*)BetaRaw_handle)[0].shape);
  void* kernel_NegAExp_shape = (((DLTensor*)NegAExp_handle)[0].shape);
  void* kernel_DtBias_shape = (((DLTensor*)DtBias_handle)[0].shape);
  void* kernel_State_shape = (((DLTensor*)State_handle)[0].shape);
  void* kernel_NormW_shape = (((DLTensor*)NormW_handle)[0].shape);
  void* kernel_Z_shape = (((DLTensor*)Z_handle)[0].shape);
  void* kernel_NewState_shape = (((DLTensor*)NewState_handle)[0].shape);
  void* kernel_Output_shape = (((DLTensor*)Output_handle)[0].shape);
  if (!(((((DLTensor*)QKV_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input QKV ndim mismatch, expected 2", (long long)((((DLTensor*)QKV_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_QKV_strides = (((DLTensor*)QKV_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)QKV_handle)[0].device.device_id);
  void* QKV = (((DLTensor*)QKV_handle)[0].data);
  if (!(((((DLTensor*)Alpha_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha ndim mismatch, expected 2", (long long)((((DLTensor*)Alpha_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_Alpha_strides = (((DLTensor*)Alpha_handle)[0].strides);
  void* Alpha = (((DLTensor*)Alpha_handle)[0].data);
  if (!(((((DLTensor*)BetaRaw_handle)[0].ndim) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw ndim mismatch, expected 2", (long long)((((DLTensor*)BetaRaw_handle)[0].ndim)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_BetaRaw_strides = (((DLTensor*)BetaRaw_handle)[0].strides);
  void* BetaRaw = (((DLTensor*)BetaRaw_handle)[0].data);
  if (!(((((DLTensor*)NegAExp_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NegAExp ndim mismatch, expected 1", (long long)((((DLTensor*)NegAExp_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_NegAExp_strides = (((DLTensor*)NegAExp_handle)[0].strides);
  void* NegAExp = (((DLTensor*)NegAExp_handle)[0].data);
  if (!(((((DLTensor*)DtBias_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input DtBias ndim mismatch, expected 1", (long long)((((DLTensor*)DtBias_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_DtBias_strides = (((DLTensor*)DtBias_handle)[0].strides);
  void* DtBias = (((DLTensor*)DtBias_handle)[0].data);
  if (!(((((DLTensor*)State_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State ndim mismatch, expected 4", (long long)((((DLTensor*)State_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_State_strides = (((DLTensor*)State_handle)[0].strides);
  void* State = (((DLTensor*)State_handle)[0].data);
  if (!(((((DLTensor*)NormW_handle)[0].ndim) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NormW ndim mismatch, expected 1", (long long)((((DLTensor*)NormW_handle)[0].ndim)), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_NormW_strides = (((DLTensor*)NormW_handle)[0].strides);
  void* NormW = (((DLTensor*)NormW_handle)[0].data);
  if (!(((((DLTensor*)Z_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z ndim mismatch, expected 3", (long long)((((DLTensor*)Z_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_Z_strides = (((DLTensor*)Z_handle)[0].strides);
  void* Z = (((DLTensor*)Z_handle)[0].data);
  if (!(((((DLTensor*)NewState_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState ndim mismatch, expected 4", (long long)((((DLTensor*)NewState_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_NewState_strides = (((DLTensor*)NewState_handle)[0].strides);
  void* NewState = (((DLTensor*)NewState_handle)[0].data);
  if (!(((((DLTensor*)Output_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output ndim mismatch, expected 3", (long long)((((DLTensor*)Output_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_Output_strides = (((DLTensor*)Output_handle)[0].strides);
  void* Output = (((DLTensor*)Output_handle)[0].data);
  if (!(((((((DLTensor*)QKV_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)QKV_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)QKV_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input QKV dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_QKV_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input QKV shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_QKV_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_QKV_shape)[1]) == 5120))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input QKV shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_QKV_shape)[1])), (long long)(5120));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((kernel_QKV_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)kernel_QKV_strides)[1]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((kernel_QKV_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)kernel_QKV_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input QKV strides[1] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)QKV_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input QKV byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)QKV_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)QKV_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input QKV device_type mismatch, expected cuda", (long long)((((DLTensor*)QKV_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(QKV == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input QKV data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Alpha_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Alpha_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Alpha_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Alpha dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Alpha_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Alpha_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Alpha_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Alpha_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((kernel_Alpha_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)kernel_Alpha_strides)[1]);
  }
  if (!((condval_2 == 1))) {
    int32_t condval_3;
    if ((kernel_Alpha_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)kernel_Alpha_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha strides[1] violates packed ABI constraint", (long long)(condval_3), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Alpha_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Alpha_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Alpha_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha device_id violates packed ABI constraint", (long long)((((DLTensor*)Alpha_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Alpha_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Alpha device_type mismatch, expected cuda", (long long)((((DLTensor*)Alpha_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Alpha == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Alpha data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)BetaRaw_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)BetaRaw_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)BetaRaw_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input BetaRaw dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_BetaRaw_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_BetaRaw_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_BetaRaw_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_BetaRaw_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((kernel_BetaRaw_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)kernel_BetaRaw_strides)[1]);
  }
  if (!((condval_4 == 1))) {
    int32_t condval_5;
    if ((kernel_BetaRaw_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)kernel_BetaRaw_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw strides[1] violates packed ABI constraint", (long long)(condval_5), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)BetaRaw_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)BetaRaw_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)BetaRaw_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw device_id violates packed ABI constraint", (long long)((((DLTensor*)BetaRaw_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)BetaRaw_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input BetaRaw device_type mismatch, expected cuda", (long long)((((DLTensor*)BetaRaw_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(BetaRaw == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input BetaRaw data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)NegAExp_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)NegAExp_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)NegAExp_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NegAExp dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_NegAExp_shape)[0]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NegAExp shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_NegAExp_shape)[0])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((kernel_NegAExp_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)kernel_NegAExp_strides)[0]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((kernel_NegAExp_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)kernel_NegAExp_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NegAExp strides[0] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)NegAExp_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NegAExp byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)NegAExp_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)NegAExp_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NegAExp device_id violates packed ABI constraint", (long long)((((DLTensor*)NegAExp_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)NegAExp_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NegAExp device_type mismatch, expected cuda", (long long)((((DLTensor*)NegAExp_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(NegAExp == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NegAExp data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)DtBias_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)DtBias_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)DtBias_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input DtBias dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_DtBias_shape)[0]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input DtBias shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_DtBias_shape)[0])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((kernel_DtBias_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)kernel_DtBias_strides)[0]);
  }
  if (!((condval_8 == 1))) {
    int32_t condval_9;
    if ((kernel_DtBias_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)kernel_DtBias_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input DtBias strides[0] violates packed ABI constraint", (long long)(condval_9), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)DtBias_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input DtBias byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)DtBias_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)DtBias_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input DtBias device_id violates packed ABI constraint", (long long)((((DLTensor*)DtBias_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)DtBias_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input DtBias device_type mismatch, expected cuda", (long long)((((DLTensor*)DtBias_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(DtBias == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input DtBias data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)State_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)State_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)State_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input State dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_State_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_State_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_State_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_State_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_State_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_State_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_State_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_State_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((kernel_State_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)kernel_State_strides)[3]);
  }
  if (!((condval_10 == 1))) {
    int32_t condval_11;
    if ((kernel_State_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)kernel_State_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State strides[3] violates packed ABI constraint", (long long)(condval_11), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((kernel_State_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)kernel_State_strides)[2]);
  }
  if (!((condval_12 == 128))) {
    int32_t condval_13;
    if ((kernel_State_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)kernel_State_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State strides[2] violates packed ABI constraint", (long long)(condval_13), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((kernel_State_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)kernel_State_strides)[1]);
  }
  if (!((condval_14 == 16384))) {
    int32_t condval_15;
    if ((kernel_State_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)kernel_State_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State strides[1] violates packed ABI constraint", (long long)(condval_15), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)State_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)State_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)State_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State device_id violates packed ABI constraint", (long long)((((DLTensor*)State_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)State_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input State device_type mismatch, expected cuda", (long long)((((DLTensor*)State_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(State == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input State data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)NormW_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)NormW_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)NormW_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NormW dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_NormW_shape)[0]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NormW shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_NormW_shape)[0])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((kernel_NormW_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)kernel_NormW_strides)[0]);
  }
  if (!((condval_16 == 1))) {
    int32_t condval_17;
    if ((kernel_NormW_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)kernel_NormW_strides)[0]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NormW strides[0] violates packed ABI constraint", (long long)(condval_17), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)NormW_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NormW byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)NormW_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)NormW_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NormW device_id violates packed ABI constraint", (long long)((((DLTensor*)NormW_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)NormW_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NormW device_type mismatch, expected cuda", (long long)((((DLTensor*)NormW_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(NormW == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NormW data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Z_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Z_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Z_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Z dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Z_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Z_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Z_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Z_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Z_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Z_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_18;
  if ((kernel_Z_strides == NULL)) {
    condval_18 = 1;
  } else {
    condval_18 = ((int32_t)((int64_t*)kernel_Z_strides)[2]);
  }
  if (!((condval_18 == 1))) {
    int32_t condval_19;
    if ((kernel_Z_strides == NULL)) {
      condval_19 = 1;
    } else {
      condval_19 = ((int32_t)((int64_t*)kernel_Z_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z strides[2] violates packed ABI constraint", (long long)(condval_19), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_20;
  if ((kernel_Z_strides == NULL)) {
    condval_20 = 1;
  } else {
    condval_20 = ((int32_t)((int64_t*)kernel_Z_strides)[1]);
  }
  if (!((condval_20 == 128))) {
    int32_t condval_21;
    if ((kernel_Z_strides == NULL)) {
      condval_21 = 1;
    } else {
      condval_21 = ((int32_t)((int64_t*)kernel_Z_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z strides[1] violates packed ABI constraint", (long long)(condval_21), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Z_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Z_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Z_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z device_id violates packed ABI constraint", (long long)((((DLTensor*)Z_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Z_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Z device_type mismatch, expected cuda", (long long)((((DLTensor*)Z_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Z == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Z data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)NewState_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)NewState_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)NewState_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NewState dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_NewState_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_NewState_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_NewState_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_NewState_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_NewState_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_NewState_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_NewState_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_NewState_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((kernel_NewState_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)kernel_NewState_strides)[3]);
  }
  if (!((condval_22 == 1))) {
    int32_t condval_23;
    if ((kernel_NewState_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)kernel_NewState_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState strides[3] violates packed ABI constraint", (long long)(condval_23), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((kernel_NewState_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)kernel_NewState_strides)[2]);
  }
  if (!((condval_24 == 128))) {
    int32_t condval_25;
    if ((kernel_NewState_strides == NULL)) {
      condval_25 = 1;
    } else {
      condval_25 = ((int32_t)((int64_t*)kernel_NewState_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState strides[2] violates packed ABI constraint", (long long)(condval_25), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((kernel_NewState_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)kernel_NewState_strides)[1]);
  }
  if (!((condval_26 == 16384))) {
    int32_t condval_27;
    if ((kernel_NewState_strides == NULL)) {
      condval_27 = 1;
    } else {
      condval_27 = ((int32_t)((int64_t*)kernel_NewState_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState strides[1] violates packed ABI constraint", (long long)(condval_27), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)NewState_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)NewState_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)NewState_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState device_id violates packed ABI constraint", (long long)((((DLTensor*)NewState_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)NewState_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input NewState device_type mismatch, expected cuda", (long long)((((DLTensor*)NewState_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(NewState == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input NewState data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Output_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Output_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Output_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Output dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Output_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Output_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Output_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Output_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Output_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Output_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_28;
  if ((kernel_Output_strides == NULL)) {
    condval_28 = 1;
  } else {
    condval_28 = ((int32_t)((int64_t*)kernel_Output_strides)[2]);
  }
  if (!((condval_28 == 1))) {
    int32_t condval_29;
    if ((kernel_Output_strides == NULL)) {
      condval_29 = 1;
    } else {
      condval_29 = ((int32_t)((int64_t*)kernel_Output_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output strides[2] violates packed ABI constraint", (long long)(condval_29), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_30;
  if ((kernel_Output_strides == NULL)) {
    condval_30 = 1;
  } else {
    condval_30 = ((int32_t)((int64_t*)kernel_Output_strides)[1]);
  }
  if (!((condval_30 == 128))) {
    int32_t condval_31;
    if ((kernel_Output_strides == NULL)) {
      condval_31 = 1;
    } else {
      condval_31 = ((int32_t)((int64_t*)kernel_Output_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output strides[1] violates packed ABI constraint", (long long)(condval_31), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Output_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Output_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_handle)[0].device.device_id) == (((DLTensor*)QKV_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output device_id violates packed ABI constraint", (long long)((((DLTensor*)Output_handle)[0].device.device_id)), (long long)((((DLTensor*)QKV_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output device_type mismatch, expected cuda", (long long)((((DLTensor*)Output_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Output == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Output data pointer is NULL");
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
  if (Alpha == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = Alpha;
  if (BetaRaw == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = BetaRaw;
  if (DtBias == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = DtBias;
  if (NegAExp == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = NegAExp;
  if (NewState == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = NewState;
  if (NormW == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = NormW;
  if (Output == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_ptr) = Output;
  if (QKV == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_ptr) = QKV;
  if (State == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_ptr) = State;
  if (Z == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_ptr) = Z;
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)24);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)30000);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = (int64_t)0;
  if (kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "kernel_kernel", &kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 16, &result_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return __tvm_ffi_kernel(self, args, num_args, result);
}
