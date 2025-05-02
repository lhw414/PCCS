// Build command : hexagon-clang -mv68 -O2 -G0 -shared -fPIC -mhvx -mhvx-length=128B -I$HEXAGON_SDK_ROOT/tools/idl -I$HEXAGON_SDK_ROOT/incs -I$HEXAGON_SDK_ROOT/incs/stddef corunkernel.c -o corunkernel.so

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#include "HAP_perf.h"      
#include "HAP_farf.h"

#define REP2(S)   S; S
#define REP4(S)   REP2(S); REP2(S)
#define REP8(S)   REP4(S); REP4(S)
#define REP16(S)  REP8(S); REP8(S)
#define REP32(S)  REP16(S); REP16(S)
#define REP64(S)  REP32(S); REP32(S)
#define REP128(S) REP64(S); REP64(S)
#define REP256(S) REP128(S); REP128(S)

static inline HVX_Vector hvx_add_vec(HVX_Vector a, HVX_Vector b)
{
    return Q6_Vh_vadd_VhVh(a, b);
}

static void hvx_mem_bw(uint8_t* buf, uint64_t nbytes, uint64_t ntrials)
{
    const uint64_t VEC_BYTES = 128;
    HVX_Vector alpha = Q6_V_vsplat_R(0x11);

    for (uint64_t t = 0; t < ntrials; ++t) {
        for (uint64_t base = 0; base < nbytes; base += VEC_BYTES) {
            HVX_Vector* vp = (HVX_Vector*)(buf + base);
            HVX_Vector v   = *vp;
            v = hvx_add_vec(v, alpha);  
            *vp = v;
        }
    }
}

int main(void)
{
    const uint64_t TOTAL_BYTES = 1ULL << 28;      
    const uint64_t NTRIALS     = 600;

    uint8_t* A;
    posix_memalign((void**)&A, 128, TOTAL_BYTES);
    memset(A, 1, TOTAL_BYTES);

    double t0 = HAP_perf_get_time_us() * 1e-6;    
    hvx_mem_bw(A, TOTAL_BYTES, NTRIALS);
    double t1 = HAP_perf_get_time_us() * 1e-6;

    uint64_t total_bytes = NTRIALS * TOTAL_BYTES * 2;
    double bw = total_bytes / (t1 - t0) / (1024.0*1024.0*1024.0);

    FARF(ALWAYS, "BW result: %.2f GiB/s  (256 MiB × %llu trials, R+W)",
         bw, (unsigned long long)NTRIALS);

    free(A);
    return 0;
}
