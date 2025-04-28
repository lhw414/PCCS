/* ──────────── corun_kernel.cl ──────────── */
/*  OpenCL kernel that mirrors the CUDA block_stride kernel          */
/*  Build example:                                                   */
/*     clang -cl-std=CL2.0 -emit-llvm -c corun_kernel.cl             */
/*  or through clBuildProgram from host at run-time                  */

#define REP2(S)        S ;        S
#define REP4(S)   REP2(S);   REP2(S)
#define REP8(S)   REP4(S);   REP4(S)
#define REP16(S)  REP8(S);   REP8(S)
#define REP32(S)  REP16(S);  REP16(S)
#define REP64(S)  REP32(S);  REP32(S)
#define REP128(S) REP64(S);  REP64(S)
#define REP256(S) REP128(S); REP128(S)
#define REP384(S) REP256(S); REP128(S)
#define REP512(S) REP256(S); REP256(S)
#define REP768(S) REP512(S); REP256(S)
#define REP1024(S) REP512(S); REP512(S)
#define REP1536(S) REP1024(S); REP512(S)
#define REP2048(S) REP1024(S); REP1024(S)
#define REP3072(S) REP2048(S); REP1024(S)
#define REP4096(S) REP2048(S); REP2048(S)
#define REP6144(S) REP4096(S); REP2048(S)
#define REP8192(S) REP4096(S); REP4096(S)

#define KERNEL2(a,b,c)   ((a) = (a)*(b) + (c))
#define KERNEL1(a,b,c)   ((a) = (b) + (c))

__kernel void block_stride(const ulong ntrials,
                           const ulong nsize,
                           __global float *A)
{
    const ulong gsize = get_global_size(0);
    const ulong gid   = get_global_id(0);

    ulong elem_per_thr = (nsize + (gsize - 1)) / gsize;
    ulong start_idx    = gid;
    ulong end_idx      = start_idx + elem_per_thr * gsize;
    if (start_idx > nsize) start_idx = nsize;
    if (end_idx   > nsize) end_idx   = nsize;

    float alpha = 0.5f;

    for (ulong j = 0; j < ntrials; ++j) {
        for (ulong i = start_idx; i < end_idx; i += gsize) {
            float beta = 0.8f;

            REP256(KERNEL2(beta, A[i], alpha));

            A[i] = beta;
        }
        alpha *= (1.0f - 1.0e-8f);
    }
}
