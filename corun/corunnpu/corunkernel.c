#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <inttypes.h>
#include <math.h>

// HVX 128-byte wide vector of float
typedef float vfloat32 __attribute__((vector_size(128)));

#define REP2(S)    S; S
#define REP4(S)    REP2(S); REP2(S)
#define REP8(S)    REP4(S); REP4(S)
#define REP16(S)   REP8(S); REP8(S)
#define REP32(S)   REP16(S); REP16(S)
#define REP64(S)   REP32(S); REP32(S)
#define REP128(S)  REP64(S); REP64(S)
#define REP256(S)  REP128(S); REP128(S)
#define REP512(S)  REP256(S); REP256(S)

static inline double now_s() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec*1e-6;
}

void hvx_bench(float *A, uint64_t nsize, uint64_t ntrials) {
    const int bytes_per_elem = sizeof(float);
    const int mem_accesses_per_elem = 2;
    const uint64_t vec_elems = 128 / sizeof(float); // 32
    uint64_t chunks = (nsize + vec_elems-1) / vec_elems;

    for (uint64_t t = 0; t < ntrials; ++t) {
        // alpha 갱신 (scalar)
        float alpha = 0.5f * powf(1.0f - 1e-8f, (float)t);
        // vector broadcast
        vfloat32 alpha_vec = { [0 ... 31] = alpha };

        for (uint64_t v = 0; v < chunks; ++v) {
            size_t base = v * vec_elems;
            if (base + vec_elems > nsize) break;

            // 로드
            vfloat32 va = *(vfloat32*)&A[base];

            // 1 flop 예시: a = b + c
            //va = va + alpha_vec;

            // 2 flop 예시: a = a*b + c
            //va = va * alpha_vec + alpha_vec;

            // 256 flop 예시:
            REP256( va = va * alpha_vec + alpha_vec; )

            // 저장
            *(vfloat32*)&A[base] = va;
        }
    }
}

int main(){
    const uint64_t TSIZE = (1ULL<<28);        // bytes
    const uint64_t nsize = TSIZE / sizeof(float);
    float *A;
    posix_memalign((void**)&A, 128, TSIZE);
    for (uint64_t i = 0; i < nsize; ++i) A[i] = 1.0f;

    uint64_t ntrials = /* OpenCL/CUDA 쪽과 동일하게 계산 */;
    double st = now_s();
    hvx_bench(A, nsize, ntrials);
    double en = now_s();

    double secs = en - st;
    uint64_t total_bytes = ntrials * nsize * sizeof(float) * 2;
    double bw = total_bytes / secs / (1024.0*1024.0*1024.0);
    printf("BW = %.3f GB/s\n", bw);
    return 0;
}
