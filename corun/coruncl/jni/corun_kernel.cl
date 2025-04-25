#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define REP2(S)    S; S
#define REP4(S)    REP2(S); REP2(S)
#define REP8(S)    REP4(S); REP4(S)
#define REP16(S)   REP8(S); REP8(S)
#define REP32(S)   REP16(S); REP16(S)
#define REP64(S)   REP32(S); REP32(S)
#define REP128(S)  REP64(S); REP64(S)
#define REP256(S)  REP128(S); REP128(S)
#define REP512(S)  REP256(S); REP256(S)

#define KERNEL1(a,b,c)  ((a) = (b) + (c))
#define KERNEL2(a,b,c)  ((a) = (a) * (b) + (c))

__kernel void block_stride(
    __global float* A,
    const ulong     ntrials,
    const ulong     nsize,
    __global int*   bytes_per_elem,
    __global int*   mem_accesses_per_elem
) {
    if (get_global_id(0) == 0) {
        *bytes_per_elem        = sizeof(float);
        *mem_accesses_per_elem = 2;
    }

    size_t global_id  = get_global_id(0);
    size_t local_id   = get_local_id(0);
    size_t local_sz   = get_local_size(0);
    size_t group_id   = get_group_id(0);
    size_t total_thr  = get_global_size(0);
    size_t elems_thr  = (nsize + total_thr - 1) / total_thr;
    size_t blk_offset = group_id * local_sz;

    size_t start_idx  = blk_offset + local_id;
    size_t end_idx    = start_idx + elems_thr * total_thr;
    if (start_idx > nsize) start_idx = nsize;
    if (end_idx   > nsize) end_idx   = nsize;
    size_t stride    = total_thr;

    float alpha = 0.5f;
    for (ulong j = 0; j < ntrials; ++j) {
        for (size_t i = start_idx; i < end_idx; i += stride) {
            float beta = 0.8f;
            // add 1 flop
            // KERNEL1(beta, A[i], alpha);
            // add 2 flop
            // KERNEL2(beta, A[i], alpha);
            // add 4 flop
            // REP2(KERNEL2(beta, A[i], alpha));
            // add 8 flop
            // REP4(KERNEL2(beta, A[i], alpha));
            // add 16 flop
            // REP8(KERNEL2(beta, A[i], alpha));
            // add 32 flop
            // REP16(KERNEL2(beta, A[i], alpha));
            // add 64 flop
            // REP32(KERNEL2(beta, A[i], alpha));
            // add 128 flop
            // REP64(KERNEL2(beta, A[i], alpha));
            // add 256 flop
            // REP128(KERNEL2(beta, A[i], alpha));
            // add 512 flop
            // REP256(KERNEL2(beta, A[i], alpha));
            // add 1024 flop
            REP512(KERNEL2(beta, A[i], alpha));

            A[i] = beta;
        }
        alpha *= (1.0f - 1e-8f);
    }
}