#include <stdio.h>
#include <riscv_vector.h>

void vector_add_test() {
    float a[] = {1.0, 2.0, 3.0, 4.0};
    float b[] = {5.0, 6.0, 7.0, 8.0};
    float res[4];

    size_t vl = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);
    vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
    __riscv_vse32_v_f32m1(res, vc, vl);

    printf("Result: %.1f, %.1f, %.1f, %.1f\n", res[0], res[1], res[2], res[3]);
}

int main() {
    vector_add_test();
    return 0;
}