#include "../yami.h"
#include <cstdio>


static void assert_equals(const yami_tensor *uut, const yami_tensor *target) {
    for (size_t i = 0; i < uut->ne; ++i) {
        YAMI_ASSERT(uut->data[i] == target->data[i]);
    }
}

static void test_2x2() {
    uint32_t dims[] = {2, 2};

    yami_tensor a(2, dims, "a");
    yami_tensor b(2, dims, "b");
    yami_tensor c(2, dims, "c");
    yami_tensor res(2, dims, "res");

    for (size_t i = 0; i < a.ne; ++i) {
       a.data[i] = 1 + i;
       b.data[i] = i % 2 == 0 ? 2 : 3;
    }
    c.data[0] = 6;
    c.data[1] = 9;
    c.data[2] = 14;
    c.data[3] = 21;

    yami_mat_mul(&res, &a, &b);

    assert_equals(&res, &c);
    printf("-> test_2x2: OK\n");
}

static void test_3x2_2x4() {
    uint32_t dims_a[] = {3, 2};
    uint32_t dims_b[] = {2, 4};
    uint32_t dims_res[] = {3, 4};

    yami_tensor a(2, dims_a, "a");
    yami_tensor b(2, dims_b, "b");
    yami_tensor c(2, dims_res, "c");
    yami_tensor res(2, dims_res, "res");

    a.data[0] = 1;
    a.data[1] = -2;
    a.data[2] = -3;
    a.data[3] = 5;
    a.data[4] = 4;
    a.data[5] = 6;

    b.data[0] = -0.5;
    b.data[1] = 1;
    b.data[2] = 0;
    b.data[3] = 7;
    b.data[4] = 4;
    b.data[5] = -12;
    b.data[6] = 10;
    b.data[7] = 2;

    res.data[0] = -8.5;
    res.data[1] = 25;
    res.data[2] = -20;
    res.data[3] = 3;
    res.data[4] = 21.5;
    res.data[5] = -63;
    res.data[6] = 50;
    res.data[7] = -11;
    res.data[8] = 22;
    res.data[9] = -68;
    res.data[10] = 60;
    res.data[11] = 40;

    yami_mat_mul(&c, &a, &b);

    assert_equals(&c, &res);
    printf("-> test_3x2_2x4: OK\n");
}

static void test_transpose_3x2() {
    uint32_t dims_a[] = {2, 3};
    uint32_t dims_res[] = {3, 2};

    yami_tensor a(2, dims_a, "a");
    yami_tensor res(2, dims_res, "res");
    a.data[0] = 1;
    a.data[1] = 3;
    a.data[2] = 5;
    a.data[3] = 2;
    a.data[4] = 4;
    a.data[5] = 6;

    res.data[0] = 1;
    res.data[1] = 2;
    res.data[2] = 3;
    res.data[3] = 4;
    res.data[4] = 5;
    res.data[5] = 6;

    yami_transpose(&a);

    assert_equals(&a, &res);
    printf("-> test_transpose_3x2: OK\n");
}

int main() {
    test_2x2();
    test_3x2_2x4();
    test_transpose_3x2();
    return 0;
}