LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE        := OpenCL
LOCAL_SRC_FILES     := prebuilt/arm64-v8a/libOpenCL.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../include   # cl.h 헤더 위치
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE       := corun_kernel
LOCAL_SRC_FILES    := host.cpp
LOCAL_C_INCLUDES   := $(LOCAL_PATH)/../include       # cl.h 헤더 위치

LOCAL_SHARED_LIBRARIES := OpenCL

LOCAL_LDLIBS       := -llog
include $(BUILD_EXECUTABLE)
