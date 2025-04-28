# 빌드할 ABI
APP_ABI            := arm64-v8a
# 최소 플랫폼 레벨. Android 7.0(API24)는 OpenMP 지원 libomp 포함
APP_PLATFORM       := android-24
# STL 필요 없으면 비워두고, OpenMP 지원
APP_STL            := none
APP_CFLAGS         := -O3 -fopenmp
APP_CPPFLAGS       := -O3 -fopenmp
# libomp (libgomp 아님!) 을 링크
APP_LDFLAGS        := -fopenmp
