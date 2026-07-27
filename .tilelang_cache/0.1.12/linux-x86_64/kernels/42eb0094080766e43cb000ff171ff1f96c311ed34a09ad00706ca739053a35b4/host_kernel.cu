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
  TL_ALIGN(128) TVMFFIAny stack[15];
  void* stack_ffi_any = stack;
  if (!((num_args == 8))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel: num_args should be 8", (long long)(num_args), (long long)(8));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel: args pointer is NULL");
    return -1;
  }
  int32_t Q_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((Q_handle_type_index == 0) || (Q_handle_type_index == 4)) || (Q_handle_type_index == 7)) || (64 <= Q_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Q expected pointer or tensor handle");
    return -1;
  }
  int32_t K_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((K_handle_type_index == 0) || (K_handle_type_index == 4)) || (K_handle_type_index == 7)) || (64 <= K_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input K expected pointer or tensor handle");
    return -1;
  }
  int32_t V_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((V_handle_type_index == 0) || (V_handle_type_index == 4)) || (V_handle_type_index == 7)) || (64 <= V_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input V expected pointer or tensor handle");
    return -1;
  }
  int32_t G_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((G_handle_type_index == 0) || (G_handle_type_index == 4)) || (G_handle_type_index == 7)) || (64 <= G_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input G expected pointer or tensor handle");
    return -1;
  }
  int32_t Beta_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((Beta_handle_type_index == 0) || (Beta_handle_type_index == 4)) || (Beta_handle_type_index == 7)) || (64 <= Beta_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Beta expected pointer or tensor handle");
    return -1;
  }
  int32_t InitState_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((InitState_handle_type_index == 0) || (InitState_handle_type_index == 4)) || (InitState_handle_type_index == 7)) || (64 <= InitState_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input InitState expected pointer or tensor handle");
    return -1;
  }
  int32_t Output_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((Output_handle_type_index == 0) || (Output_handle_type_index == 4)) || (Output_handle_type_index == 7)) || (64 <= Output_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Output expected pointer or tensor handle");
    return -1;
  }
  int32_t FinalState_handle_type_index = (((TVMFFIAny*)args)[7].type_index);
  if (!(((((FinalState_handle_type_index == 0) || (FinalState_handle_type_index == 4)) || (FinalState_handle_type_index == 7)) || (64 <= FinalState_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input FinalState expected pointer or tensor handle");
    return -1;
  }
  void* Q_handle = ((Q_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* K_handle = ((K_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* V_handle = ((V_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* G_handle = ((G_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* Beta_handle = ((Beta_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* InitState_handle = ((InitState_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  void* Output_handle = ((Output_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  void* FinalState_handle = ((FinalState_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + 24)) : (((TVMFFIAny*)args)[7].v_ptr));
  bool kernel_Q_is_null = (Q_handle == NULL);
  if (!(!kernel_Q_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.Q is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_K_is_null = (K_handle == NULL);
  if (!(!kernel_K_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.K is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_V_is_null = (V_handle == NULL);
  if (!(!kernel_V_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.V is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_G_is_null = (G_handle == NULL);
  if (!(!kernel_G_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.G is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_Beta_is_null = (Beta_handle == NULL);
  if (!(!kernel_Beta_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.Beta is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_InitState_is_null = (InitState_handle == NULL);
  if (!(!kernel_InitState_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.InitState is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_Output_is_null = (Output_handle == NULL);
  if (!(!kernel_Output_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.Output is expected to have non-NULL pointer");
    return -1;
  }
  bool kernel_FinalState_is_null = (FinalState_handle == NULL);
  if (!(!kernel_FinalState_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel.FinalState is expected to have non-NULL pointer");
    return -1;
  }
  void* kernel_Q_shape = (((DLTensor*)Q_handle)[0].shape);
  void* kernel_K_shape = (((DLTensor*)K_handle)[0].shape);
  void* kernel_V_shape = (((DLTensor*)V_handle)[0].shape);
  void* kernel_G_shape = (((DLTensor*)G_handle)[0].shape);
  void* kernel_Beta_shape = (((DLTensor*)Beta_handle)[0].shape);
  void* kernel_InitState_shape = (((DLTensor*)InitState_handle)[0].shape);
  void* kernel_Output_shape = (((DLTensor*)Output_handle)[0].shape);
  void* kernel_FinalState_shape = (((DLTensor*)FinalState_handle)[0].shape);
  if (!(((((DLTensor*)Q_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q ndim mismatch, expected 4", (long long)((((DLTensor*)Q_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_Q_strides = (((DLTensor*)Q_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)Q_handle)[0].device.device_id);
  void* Q = (((DLTensor*)Q_handle)[0].data);
  if (!(((((DLTensor*)K_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K ndim mismatch, expected 4", (long long)((((DLTensor*)K_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_K_strides = (((DLTensor*)K_handle)[0].strides);
  void* K = (((DLTensor*)K_handle)[0].data);
  if (!(((((DLTensor*)V_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V ndim mismatch, expected 4", (long long)((((DLTensor*)V_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_V_strides = (((DLTensor*)V_handle)[0].strides);
  void* V = (((DLTensor*)V_handle)[0].data);
  if (!(((((DLTensor*)G_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G ndim mismatch, expected 3", (long long)((((DLTensor*)G_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_G_strides = (((DLTensor*)G_handle)[0].strides);
  void* G = (((DLTensor*)G_handle)[0].data);
  if (!(((((DLTensor*)Beta_handle)[0].ndim) == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta ndim mismatch, expected 3", (long long)((((DLTensor*)Beta_handle)[0].ndim)), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_Beta_strides = (((DLTensor*)Beta_handle)[0].strides);
  void* Beta = (((DLTensor*)Beta_handle)[0].data);
  if (!(((((DLTensor*)InitState_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState ndim mismatch, expected 4", (long long)((((DLTensor*)InitState_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_InitState_strides = (((DLTensor*)InitState_handle)[0].strides);
  void* InitState = (((DLTensor*)InitState_handle)[0].data);
  if (!(((((DLTensor*)Output_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output ndim mismatch, expected 4", (long long)((((DLTensor*)Output_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_Output_strides = (((DLTensor*)Output_handle)[0].strides);
  void* Output = (((DLTensor*)Output_handle)[0].data);
  if (!(((((DLTensor*)FinalState_handle)[0].ndim) == 4))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState ndim mismatch, expected 4", (long long)((((DLTensor*)FinalState_handle)[0].ndim)), (long long)(4));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  void* kernel_FinalState_strides = (((DLTensor*)FinalState_handle)[0].strides);
  void* FinalState = (((DLTensor*)FinalState_handle)[0].data);
  if (!(((((((DLTensor*)Q_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)Q_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)Q_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Q dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Q_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Q_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Q_shape)[1]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Q_shape)[1])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Q_shape)[2]) == 8))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Q_shape)[2])), (long long)(8));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Q_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Q_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval;
  if ((kernel_Q_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)kernel_Q_strides)[3]);
  }
  if (!((condval == 1))) {
    int32_t condval_1;
    if ((kernel_Q_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)kernel_Q_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q strides[3] violates packed ABI constraint", (long long)(condval_1), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_2;
  if ((kernel_Q_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)kernel_Q_strides)[2]);
  }
  if (!((condval_2 == 128))) {
    int32_t condval_3;
    if ((kernel_Q_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)kernel_Q_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q strides[2] violates packed ABI constraint", (long long)(condval_3), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_4;
  if ((kernel_Q_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)kernel_Q_strides)[1]);
  }
  if (!((condval_4 == 1024))) {
    int32_t condval_5;
    if ((kernel_Q_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)kernel_Q_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q strides[1] violates packed ABI constraint", (long long)(condval_5), (long long)(1024));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Q_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Q_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Q_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Q device_type mismatch, expected cuda", (long long)((((DLTensor*)Q_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Q == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Q data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)K_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)K_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)K_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input K dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_K_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_K_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_K_shape)[1]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_K_shape)[1])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_K_shape)[2]) == 8))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_K_shape)[2])), (long long)(8));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_K_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_K_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_6;
  if ((kernel_K_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)kernel_K_strides)[3]);
  }
  if (!((condval_6 == 1))) {
    int32_t condval_7;
    if ((kernel_K_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)kernel_K_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K strides[3] violates packed ABI constraint", (long long)(condval_7), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_8;
  if ((kernel_K_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)kernel_K_strides)[2]);
  }
  if (!((condval_8 == 128))) {
    int32_t condval_9;
    if ((kernel_K_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)kernel_K_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K strides[2] violates packed ABI constraint", (long long)(condval_9), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_10;
  if ((kernel_K_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)kernel_K_strides)[1]);
  }
  if (!((condval_10 == 1024))) {
    int32_t condval_11;
    if ((kernel_K_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)kernel_K_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K strides[1] violates packed ABI constraint", (long long)(condval_11), (long long)(1024));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)K_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)K_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K device_id violates packed ABI constraint", (long long)((((DLTensor*)K_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)K_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input K device_type mismatch, expected cuda", (long long)((((DLTensor*)K_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(K == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input K data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)V_handle)[0].dtype.code) == (uint8_t)4) && ((((DLTensor*)V_handle)[0].dtype.bits) == (uint8_t)16)) && ((((DLTensor*)V_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input V dtype mismatch, expected bfloat16");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_V_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_V_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_V_shape)[1]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_V_shape)[1])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_V_shape)[2]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_V_shape)[2])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_V_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_V_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_12;
  if ((kernel_V_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)kernel_V_strides)[3]);
  }
  if (!((condval_12 == 1))) {
    int32_t condval_13;
    if ((kernel_V_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)kernel_V_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V strides[3] violates packed ABI constraint", (long long)(condval_13), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_14;
  if ((kernel_V_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)kernel_V_strides)[2]);
  }
  if (!((condval_14 == 128))) {
    int32_t condval_15;
    if ((kernel_V_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)kernel_V_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V strides[2] violates packed ABI constraint", (long long)(condval_15), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_16;
  if ((kernel_V_strides == NULL)) {
    condval_16 = 1;
  } else {
    condval_16 = ((int32_t)((int64_t*)kernel_V_strides)[1]);
  }
  if (!((condval_16 == 3072))) {
    int32_t condval_17;
    if ((kernel_V_strides == NULL)) {
      condval_17 = 1;
    } else {
      condval_17 = ((int32_t)((int64_t*)kernel_V_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V strides[1] violates packed ABI constraint", (long long)(condval_17), (long long)(3072));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)V_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)V_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V device_id violates packed ABI constraint", (long long)((((DLTensor*)V_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)V_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input V device_type mismatch, expected cuda", (long long)((((DLTensor*)V_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(V == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input V data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)G_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)G_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)G_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input G dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_G_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_G_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_G_shape)[1]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_G_shape)[1])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_G_shape)[2]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_G_shape)[2])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_18;
  if ((kernel_G_strides == NULL)) {
    condval_18 = 1;
  } else {
    condval_18 = ((int32_t)((int64_t*)kernel_G_strides)[2]);
  }
  if (!((condval_18 == 1))) {
    int32_t condval_19;
    if ((kernel_G_strides == NULL)) {
      condval_19 = 1;
    } else {
      condval_19 = ((int32_t)((int64_t*)kernel_G_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G strides[2] violates packed ABI constraint", (long long)(condval_19), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_20;
  if ((kernel_G_strides == NULL)) {
    condval_20 = 1;
  } else {
    condval_20 = ((int32_t)((int64_t*)kernel_G_strides)[1]);
  }
  if (!((condval_20 == 24))) {
    int32_t condval_21;
    if ((kernel_G_strides == NULL)) {
      condval_21 = 1;
    } else {
      condval_21 = ((int32_t)((int64_t*)kernel_G_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G strides[1] violates packed ABI constraint", (long long)(condval_21), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)G_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)G_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)G_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G device_id violates packed ABI constraint", (long long)((((DLTensor*)G_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)G_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input G device_type mismatch, expected cuda", (long long)((((DLTensor*)G_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(G == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input G data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)Beta_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)Beta_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)Beta_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Beta dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Beta_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Beta_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Beta_shape)[1]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Beta_shape)[1])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Beta_shape)[2]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Beta_shape)[2])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_22;
  if ((kernel_Beta_strides == NULL)) {
    condval_22 = 1;
  } else {
    condval_22 = ((int32_t)((int64_t*)kernel_Beta_strides)[2]);
  }
  if (!((condval_22 == 1))) {
    int32_t condval_23;
    if ((kernel_Beta_strides == NULL)) {
      condval_23 = 1;
    } else {
      condval_23 = ((int32_t)((int64_t*)kernel_Beta_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta strides[2] violates packed ABI constraint", (long long)(condval_23), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_24;
  if ((kernel_Beta_strides == NULL)) {
    condval_24 = 1;
  } else {
    condval_24 = ((int32_t)((int64_t*)kernel_Beta_strides)[1]);
  }
  if (!((condval_24 == 24))) {
    int32_t condval_25;
    if ((kernel_Beta_strides == NULL)) {
      condval_25 = 1;
    } else {
      condval_25 = ((int32_t)((int64_t*)kernel_Beta_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta strides[1] violates packed ABI constraint", (long long)(condval_25), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Beta_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Beta_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Beta_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta device_id violates packed ABI constraint", (long long)((((DLTensor*)Beta_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Beta_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Beta device_type mismatch, expected cuda", (long long)((((DLTensor*)Beta_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(Beta == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input Beta data pointer is NULL");
    return -1;
  }
  if (!(((((((DLTensor*)InitState_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)InitState_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)InitState_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input InitState dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_InitState_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_InitState_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_InitState_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_InitState_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_InitState_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_InitState_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_InitState_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_InitState_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_26;
  if ((kernel_InitState_strides == NULL)) {
    condval_26 = 1;
  } else {
    condval_26 = ((int32_t)((int64_t*)kernel_InitState_strides)[3]);
  }
  if (!((condval_26 == 1))) {
    int32_t condval_27;
    if ((kernel_InitState_strides == NULL)) {
      condval_27 = 1;
    } else {
      condval_27 = ((int32_t)((int64_t*)kernel_InitState_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState strides[3] violates packed ABI constraint", (long long)(condval_27), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_28;
  if ((kernel_InitState_strides == NULL)) {
    condval_28 = 1;
  } else {
    condval_28 = ((int32_t)((int64_t*)kernel_InitState_strides)[2]);
  }
  if (!((condval_28 == 128))) {
    int32_t condval_29;
    if ((kernel_InitState_strides == NULL)) {
      condval_29 = 1;
    } else {
      condval_29 = ((int32_t)((int64_t*)kernel_InitState_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState strides[2] violates packed ABI constraint", (long long)(condval_29), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_30;
  if ((kernel_InitState_strides == NULL)) {
    condval_30 = 1;
  } else {
    condval_30 = ((int32_t)((int64_t*)kernel_InitState_strides)[1]);
  }
  if (!((condval_30 == 16384))) {
    int32_t condval_31;
    if ((kernel_InitState_strides == NULL)) {
      condval_31 = 1;
    } else {
      condval_31 = ((int32_t)((int64_t*)kernel_InitState_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState strides[1] violates packed ABI constraint", (long long)(condval_31), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)InitState_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)InitState_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)InitState_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState device_id violates packed ABI constraint", (long long)((((DLTensor*)InitState_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)InitState_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input InitState device_type mismatch, expected cuda", (long long)((((DLTensor*)InitState_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(InitState == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input InitState data pointer is NULL");
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
  if (!((((int32_t)((int64_t*)kernel_Output_shape)[1]) == 32))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Output_shape)[1])), (long long)(32));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Output_shape)[2]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Output_shape)[2])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_Output_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_Output_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_32;
  if ((kernel_Output_strides == NULL)) {
    condval_32 = 1;
  } else {
    condval_32 = ((int32_t)((int64_t*)kernel_Output_strides)[3]);
  }
  if (!((condval_32 == 1))) {
    int32_t condval_33;
    if ((kernel_Output_strides == NULL)) {
      condval_33 = 1;
    } else {
      condval_33 = ((int32_t)((int64_t*)kernel_Output_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output strides[3] violates packed ABI constraint", (long long)(condval_33), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_34;
  if ((kernel_Output_strides == NULL)) {
    condval_34 = 1;
  } else {
    condval_34 = ((int32_t)((int64_t*)kernel_Output_strides)[2]);
  }
  if (!((condval_34 == 128))) {
    int32_t condval_35;
    if ((kernel_Output_strides == NULL)) {
      condval_35 = 1;
    } else {
      condval_35 = ((int32_t)((int64_t*)kernel_Output_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output strides[2] violates packed ABI constraint", (long long)(condval_35), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_36;
  if ((kernel_Output_strides == NULL)) {
    condval_36 = 1;
  } else {
    condval_36 = ((int32_t)((int64_t*)kernel_Output_strides)[1]);
  }
  if (!((condval_36 == 3072))) {
    int32_t condval_37;
    if ((kernel_Output_strides == NULL)) {
      condval_37 = 1;
    } else {
      condval_37 = ((int32_t)((int64_t*)kernel_Output_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output strides[1] violates packed ABI constraint", (long long)(condval_37), (long long)(3072));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)Output_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)Output_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)Output_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input Output device_id violates packed ABI constraint", (long long)((((DLTensor*)Output_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
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
  if (!(((((((DLTensor*)FinalState_handle)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)FinalState_handle)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)FinalState_handle)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input FinalState dtype mismatch, expected float32");
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_FinalState_shape)[0]) == 1))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState shape[0] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_FinalState_shape)[0])), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_FinalState_shape)[1]) == 24))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState shape[1] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_FinalState_shape)[1])), (long long)(24));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_FinalState_shape)[2]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState shape[2] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_FinalState_shape)[2])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!((((int32_t)((int64_t*)kernel_FinalState_shape)[3]) == 128))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState shape[3] violates packed ABI constraint", (long long)(((int32_t)((int64_t*)kernel_FinalState_shape)[3])), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_38;
  if ((kernel_FinalState_strides == NULL)) {
    condval_38 = 1;
  } else {
    condval_38 = ((int32_t)((int64_t*)kernel_FinalState_strides)[3]);
  }
  if (!((condval_38 == 1))) {
    int32_t condval_39;
    if ((kernel_FinalState_strides == NULL)) {
      condval_39 = 1;
    } else {
      condval_39 = ((int32_t)((int64_t*)kernel_FinalState_strides)[3]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState strides[3] violates packed ABI constraint", (long long)(condval_39), (long long)(1));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_40;
  if ((kernel_FinalState_strides == NULL)) {
    condval_40 = 1;
  } else {
    condval_40 = ((int32_t)((int64_t*)kernel_FinalState_strides)[2]);
  }
  if (!((condval_40 == 128))) {
    int32_t condval_41;
    if ((kernel_FinalState_strides == NULL)) {
      condval_41 = 1;
    } else {
      condval_41 = ((int32_t)((int64_t*)kernel_FinalState_strides)[2]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState strides[2] violates packed ABI constraint", (long long)(condval_41), (long long)(128));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  int32_t condval_42;
  if ((kernel_FinalState_strides == NULL)) {
    condval_42 = 1;
  } else {
    condval_42 = ((int32_t)((int64_t*)kernel_FinalState_strides)[1]);
  }
  if (!((condval_42 == 16384))) {
    int32_t condval_43;
    if ((kernel_FinalState_strides == NULL)) {
      condval_43 = 1;
    } else {
      condval_43 = ((int32_t)((int64_t*)kernel_FinalState_strides)[1]);
    }
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState strides[1] violates packed ABI constraint", (long long)(condval_43), (long long)(16384));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)FinalState_handle)[0].byte_offset)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState byte_offset violates packed ABI constraint", (long long)((uint64_t)0), (long long)((((DLTensor*)FinalState_handle)[0].byte_offset)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)FinalState_handle)[0].device.device_id) == (((DLTensor*)Q_handle)[0].device.device_id)))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState device_id violates packed ABI constraint", (long long)((((DLTensor*)FinalState_handle)[0].device.device_id)), (long long)((((DLTensor*)Q_handle)[0].device.device_id)));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(((((DLTensor*)FinalState_handle)[0].device.device_type) == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "kernel kernel input FinalState device_type mismatch, expected cuda", (long long)((((DLTensor*)FinalState_handle)[0].device.device_type)), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(FinalState == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel kernel input FinalState data pointer is NULL");
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
  if (Beta == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = Beta;
  if (FinalState == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = FinalState;
  if (G == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = G;
  if (InitState == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = InitState;
  if (K == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = K;
  if (Output == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = Output;
  if (Q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_ptr) = Q;
  if (V == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_ptr) = V;
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)24);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)110848);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = (int64_t)0;
  if (kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "kernel_kernel", &kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_2;
  result_2.type_index = kTVMFFINone;
  result_2.zero_padding = 0;
  result_2.v_int64 = 0;
  if (TVMFFIFunctionCall(kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 14, &result_2) != 0) {
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
