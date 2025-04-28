LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := driver1
LOCAL_SRC_FILES := driver1.c

LOCAL_CFLAGS    += -fopenmp
LOCAL_CPPFLAGS  += -fopenmp
LOCAL_LDFLAGS   += -static-openmp

include $(BUILD_EXECUTABLE)
