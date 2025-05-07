// cpu_bw_test.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

#include "hvx_hexagon_protos.h"

#define TOTAL_BYTES   (1ULL<<28)  
#define NTRIALS       600
#define NUM_WORKERS   4            

typedef struct {
  uint8_t* buf;
  uint64_t bytes;
  uint64_t trials;
} worker_arg_t;

static void* worker(void* arg) {
  worker_arg_t* a = (worker_arg_t*)arg;

  hvx_mem_bw(a->buf, a->bytes, a->trials);
  return NULL;
}

int main() {
  uint8_t* A;
  if (posix_memalign((void**)&A, 128, TOTAL_BYTES)) {
    perror("posix_memalign");
    return 1;
  }
  memset(A, 1, TOTAL_BYTES);

  pthread_t th[NUM_WORKERS];
  worker_arg_t args[NUM_WORKERS];
  uint64_t chunk = TOTAL_BYTES / NUM_WORKERS;

  struct timeval tv0, tv1;
  gettimeofday(&tv0, NULL);

  for (int i = 0; i < NUM_WORKERS; i++) {
    args[i].buf    = A + i*chunk;
    args[i].bytes  = (i==NUM_WORKERS-1)
                      ? chunk + TOTAL_BYTES%NUM_WORKERS
                      : chunk;
    args[i].trials = NTRIALS;
    pthread_create(&th[i], NULL, worker, &args[i]);
  }
  for (int i = 0; i < NUM_WORKERS; i++) {
    pthread_join(th[i], NULL);
  }

  gettimeofday(&tv1, NULL);
  double elapsed = (tv1.tv_sec - tv0.tv_sec)
                 + (tv1.tv_usec - tv0.tv_usec) * 1e-6;

  double total_rw_bytes = (double)TOTAL_BYTES * NTRIALS * 2.0;
  double bw = total_rw_bytes / elapsed 
            / (1024.0*1024.0*1024.0);

  printf("[CPU] %d-way HVX BW: %.2f GiB/s (elapsed=%.3fs)\n",
         NUM_WORKERS, bw, elapsed);

  free(A);
  return 0;
}
