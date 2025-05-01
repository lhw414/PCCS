#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

void hvx_add(const uint8_t *a, const uint8_t *b, uint8_t *c) {
    HVX_Vector va = *(HVX_Vector *)a;
    HVX_Vector vb = *(HVX_Vector *)b;
    HVX_Vector vc = Q6_Vh_vadd_VhVh(va, vb);
    *(HVX_Vector *)c = vc;
}

int main() {
    uint8_t *a = (uint8_t *)memalign(128, 128);
    uint8_t *b = (uint8_t *)memalign(128, 128);
    uint8_t *c = (uint8_t *)memalign(128, 128);

    for (int i = 0; i < 128; i++) {
        a[i] = i;
        b[i] = 128 - i;
    }

    hvx_add(a, b, c);

    for (int i = 0; i < 16; i++) {
        printf("%3d ", c[i]);
    }
    printf("\n");

    free(a); free(b); free(c);
    return 0;
}
