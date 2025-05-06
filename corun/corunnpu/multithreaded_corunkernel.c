// Build command:
// hexagon-clang -mv68 -O2 -G0 -shared -fPIC -mhvx -mhvx-length=128B \
//   -I$HEXAGON_SDK_ROOT/tools/idl -I$HEXAGON_SDK_ROOT/incs
//   -I$HEXAGON_SDK_ROOT/incs/stddef \
//   -I$HEXAGON_SDK_ROOT/libs/common/qurt/ADSPv68MP/include \
//   -L$HEXAGON_SDK_ROOT/libs/common/qurt/ADSPv68MP/lib \
//   multithreaded_corunkernel.c -o multithreaded_corunkernel.so -lqurt
//   -lpthread

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_farf.h"
#include "HAP_perf.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#include "qurt_barrier.h"
#include "qurt_cycles.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"

#define MAX_THREADS 8

// REP macros for unrolling (from original code)
#define REP2(S) \
  S;            \
  S
#define REP4(S) \
  REP2(S);      \
  REP2(S)
#define REP8(S) \
  REP4(S);      \
  REP4(S)
#define REP16(S) \
  REP8(S);       \
  REP8(S)
#define REP32(S) \
  REP16(S);      \
  REP16(S)
#define REP64(S) \
  REP32(S);      \
  REP32(S)
#define REP128(S) \
  REP64(S);       \
  REP64(S)
#define REP256(S) \
  REP128(S);      \
  REP128(S)

typedef struct {
  uint8_t* buf;
  uint64_t nbytes;
  uint64_t ntrials;
  int thread_id;
  qurt_barrier_t* barrier;
  unsigned int hvx_lock_success;
} thread_arg_t;

static inline HVX_Vector hvx_add_vec(HVX_Vector a, HVX_Vector b) {
  return Q6_Vh_vadd_VhVh(a, b);
}

static void hvx_mem_bw_thread(void* arg) {
  thread_arg_t* params = (thread_arg_t*)arg;
  const uint64_t VEC_BYTES = 128;  // HVX vector size
  HVX_Vector alpha = Q6_V_vsplat_R(0x11);

  int lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
  if (lock_status != QURT_EOK) {
    FARF(ERROR, "Thread %d: Failed to lock HVX, error %d", params->thread_id,
         lock_status);
    params->hvx_lock_success = 0;
    qurt_barrier_wait(params->barrier);
    qurt_thread_exit(1);
    return;
  }
  params->hvx_lock_success = 1;
  FARF(HIGH, "Thread %d: HVX locked successfully.", params->thread_id);

  for (uint64_t t = 0; t < params->ntrials; ++t) {
    for (uint64_t base = 0; base < params->nbytes; base += VEC_BYTES) {
      if (base + VEC_BYTES <= params->nbytes) {
        HVX_Vector* vp = (HVX_Vector*)(params->buf + base);
        HVX_Vector v = *vp;
        v = hvx_add_vec(v, alpha);
        *vp = v;
      }
    }
  }

  qurt_hvx_unlock();
  FARF(HIGH, "Thread %d: HVX unlocked.", params->thread_id);

  qurt_barrier_wait(params->barrier);

  qurt_thread_exit(0);
}

int main(void) {
  const uint64_t TOTAL_BYTES = 1ULL << 28;
  const uint64_t NTRIALS = 600;
  int num_hvx_units = 0;
  int num_threads_to_use = 0;

  FARF(ALWAYS, "Starting multithreaded HVX memory bandwidth test...");

  num_hvx_units = qurt_hvx_get_units();
  if (num_hvx_units <= 0) {
    FARF(ERROR,
         "Failed to get HVX units or no HVX units available: %d. Defaulting to "
         "1 thread.",
         num_hvx_units);
    num_threads_to_use = 1;
  } else {
    num_threads_to_use = num_hvx_units;
    FARF(ALWAYS, "Detected %d HVX units. Will use %d threads.", num_hvx_units,
         num_threads_to_use);
  }

  if (num_threads_to_use > MAX_THREADS) {
    FARF(ALWAYS,
         "Number of HVX units (%d) exceeds MAX_THREADS (%d). Clamping to "
         "MAX_THREADS.",
         num_threads_to_use, MAX_THREADS);
    num_threads_to_use = MAX_THREADS;
  }

  uint8_t* A;
  int ret = posix_memalign((void**)&A, 128, TOTAL_BYTES);
  if (ret != 0 || A == NULL) {
    FARF(ERROR, "posix_memalign failed! Error: %d", ret);
    return -1;
  }
  memset(A, 1, TOTAL_BYTES);
  FARF(HIGH, "Memory allocated and initialized.");

  qurt_thread_t threads[MAX_THREADS];
  thread_arg_t thread_args[MAX_THREADS];
  qurt_thread_attr_t thread_attrs[MAX_THREADS];
  qurt_barrier_t barrier;

  qurt_barrier_init(&barrier, num_threads_to_use);

  uint64_t bytes_per_thread = TOTAL_BYTES / num_threads_to_use;
  uint64_t remaining_bytes = TOTAL_BYTES % num_threads_to_use;

  double t0, t1;
  unsigned long long start_cycles, end_cycles;

  FARF(HIGH, "Creating %d worker threads...", num_threads_to_use);
  start_cycles = qurt_cycles_get();
  t0 = HAP_perf_get_time_us() * 1e-6;

  for (int i = 0; i < num_threads_to_use; ++i) {
    qurt_thread_attr_init(&thread_attrs[i]);

    char thread_name[32];
    snprintf(thread_name, sizeof(thread_name), "hvx_bw_thread_%d", i);
    qurt_thread_attr_set_name(&thread_attrs[i], thread_name);

    thread_args[i].thread_id = i;
    thread_args[i].buf = A + (i * bytes_per_thread);
    thread_args[i].ntrials = NTRIALS;
    thread_args[i].barrier = &barrier;
    thread_args[i].hvx_lock_success = 0;

    if (i == num_threads_to_use - 1) {
      thread_args[i].nbytes = bytes_per_thread + remaining_bytes;
    } else {
      thread_args[i].nbytes = bytes_per_thread;
    }

    FARF(HIGH, "Thread %d gets %llu bytes, offset %llu", i,
         (unsigned long long)thread_args[i].nbytes,
         (unsigned long long)(i * bytes_per_thread));

    ret = qurt_thread_create(&threads[i], &thread_attrs[i], hvx_mem_bw_thread,
                             (void*)&thread_args[i]);
    if (ret != QURT_EOK) {
      FARF(ERROR, "Failed to create thread %d, error %d", i, ret);
      for (int k = 0; k < i; ++k) qurt_thread_join(threads[k], NULL);
      free(A);
      qurt_barrier_destroy(&barrier);
      return -1;
    }
  }

  FARF(HIGH, "All threads created. Waiting for them to complete...");

  int all_threads_succeeded = 1;
  for (int i = 0; i < num_threads_to_use; ++i) {
    int status;
    qurt_thread_join(threads[i], &status);
    if (status != 0 || thread_args[i].hvx_lock_success == 0) {
      FARF(ERROR, "Thread %d exited with status %d or HVX lock failed (%u).", i,
           status, thread_args[i].hvx_lock_success);
      all_threads_succeeded = 0;
    }
  }

  t1 = HAP_perf_get_time_us() * 1e-6;
  end_cycles = qurt_cycles_get();

  if (!all_threads_succeeded) {
    FARF(ERROR,
         "One or more threads failed. Bandwidth calculation might be "
         "inaccurate or test failed.");
  } else {
    FARF(HIGH, "All threads completed successfully.");
  }

  uint64_t total_data_processed = NTRIALS * TOTAL_BYTES * 2;
  double elapsed_time_sec = t1 - t0;
  unsigned long long elapsed_cycles = end_cycles - start_cycles;
  double bw_gib_s = 0;

  if (elapsed_time_sec > 0) {
    bw_gib_s =
        total_data_processed / elapsed_time_sec / (1024.0 * 1024.0 * 1024.0);
  }

  FARF(ALWAYS, "-----------------------------------------------------");
  FARF(ALWAYS, "Multithreaded HVX Memory Bandwidth Test Results:");
  FARF(ALWAYS, "Number of Threads: %d", num_threads_to_use);
  FARF(ALWAYS, "Total Bytes per access: %llu MiB",
       (unsigned long long)(TOTAL_BYTES / (1024 * 1024)));
  FARF(ALWAYS, "Number of Trials: %llu", (unsigned long long)NTRIALS);
  FARF(ALWAYS, "Total Data Processed (R+W): %.2f GiB",
       total_data_processed / (1024.0 * 1024.0 * 1024.0));
  FARF(ALWAYS, "Elapsed Time: %.4f seconds", elapsed_time_sec);
  FARF(ALWAYS, "Elapsed Cycles: %llu cycles", elapsed_cycles);
  FARF(ALWAYS, "Achieved Bandwidth: %.2f GiB/s", bw_gib_s);
  FARF(ALWAYS, "-----------------------------------------------------");

  free(A);
  qurt_barrier_destroy(&barrier);
  FARF(ALWAYS, "Test finished. Resources freed.");

  return 0;
}
