#!/usr/bin/env sh
set -e

ANDROID_NDK_ROOT=${ANDROID_NDK_HOME:-/opt/android/ndk}     # NDK 설치 경로
HEXAGON_SDK_ROOT=/opt/Qualcomm/Hexagon_SDK/6.0.0.2        # Hexagon SDK 설치 경로

# Hexagon-Tools 바이너리 위치
HEXAGON_TOOLS=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.7.08/Tools/bin
# Hexagon SDK 내 sysroot (HVX 헤더)
HEXAGON_SYSROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.7.08/Tools/hexagon-sdk/sysroot

# Android NDK sysroot (bionic 표준 헤더)
ANDROID_SYSROOT=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/sysroot

# cross-compiler
CC=$HEXAGON_TOOLS/hexagon-clang

# 공통 플래그
COMMON_FLAGS="
  --target=hexagon-unknown-linux-gnueabi
  -mcpu=hexagonv67
  -O3
  -std=c11
  -mhvx                  # enable HVX
  -isysroot $ANDROID_SYSROOT
"

# HVX 전용 include 경로
INCLUDE_HVX="-I$HEXAGON_SYSROOT/usr/include"

# 빌드
echo ">>> Compiling corunkernel.c → corunkernel.o"
$CC $COMMON_FLAGS $INCLUDE_HVX corunkernel.c -o corunkernel.o

echo ">>> Linking corunkernel.o → corunkernel.elf"
$CC $COMMON_FLAGS corunkernel.o -o corunkernel.elf

echo ">>> Done."
