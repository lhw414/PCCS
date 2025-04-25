#pragma OPENCL EXTENSION cl_khr_int64_base_atomics  : enable

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
    __global float*      A,                   
    const ulong           ntrials,              
    const ulong           nsize,                
    __global int*         bytes_per_elem,       
    __global int*         mem_accesses_per_elem 
) {
    if (get_global_id(0) == 0) {
        *bytes_per_elem        = sizeof(float);
        *mem_accesses_per_elem = 2;
    }

    // size_t gid           = get_global_id(0);
    // size_t total_threads = get_global_size(0);
    // size_t stride        = total_threads;
    // float alpha         = 0.5;

    size_t total_threads = get_global_size(0);
    size_t elem_per_thread = (nsize + total_threads - 1) / total_threads;
    size_t block_offset = get_local_size(0) * get_local_id(0);

    size_t start_idx = block_offset + get_local_id(0);
    size_t end_idx = start_idx + elem_per_thread * total_threads;
    size_t stride_idx = total_threads;

    float alpha = 0.5;

    for (ulong j = 0; j < ntrials; ++j) {
        for (size_t i = start_idx; i < end_idx; i += stride_idx) {
            float beta = 0.8;
            // adjust compute intensity 
            KERNEL1(beta, A[i], alpha);
            // add 2 flops
            // KERNEL2(beta, A[i], alpha);
            // add 4 flops
            // REP2(KERNEL2(beta, A[i], alpha));
            // add 8 flops
            // REP4(KERNEL2(beta, A[i], alpha));
            // add 16 flops
            // REP8(KERNEL2(beta, A[i], alpha));
            // add 32 flops
            // REP16(KERNEL2(beta, A[i], alpha));
            // add 64 flops
            // REP32(KERNEL2(beta, A[i], alpha));
            // add 128 flops
            // REP64(KERNEL2(beta, A[i], alpha));
            // add 256 flops
            // REP128(KERNEL2(beta, A[i], alpha));
            // add 512 flops
            // REP256(KERNEL2(beta, A[i], alpha));
            // add 1024 flops
            // REP512(KERNEL2(beta, A[i], alpha));

            A[i] = beta;
        }
        alpha *= (1.0 - 1e-8);
    }
}
