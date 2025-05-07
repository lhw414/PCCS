// Build command :
// hexagon-clang -mv68 -O2 -G0 -shared -fPIC \
//     -mhvx -mhvx-length=128B \
//     -I$HEXAGON_SDK_ROOT/tools/idl \
//     -I$HEXAGON_SDK_ROOT/incs \
//     -I$HEXAGON_SDK_ROOT/incs/stddef \
//     corunkernel_.c -o corunkernel_.so

#include <stdint.h>
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

static inline HVX_Vector hvx_add_vec(HVX_Vector a, HVX_Vector b) {
  return Q6_Vh_vadd_VhVh(a, b);
}

void hvx_mem_bw(uint8_t* buf, uint64_t nbytes, uint64_t ntrials) {
  const uint64_t VEC_BYTES = 128;
  HVX_Vector alpha = Q6_V_vsplat_R(0x11);
  for (uint64_t t = 0; t < ntrials; t++) {
    for (uint64_t off = 0; off < nbytes; off += VEC_BYTES) {
      HVX_Vector* vp = (HVX_Vector*)(buf + off);
      HVX_Vector v = *vp;
      v = hvx_add_vec(v, alpha);
      *vp = v;
    }
  }
}
