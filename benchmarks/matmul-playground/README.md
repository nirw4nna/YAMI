# MatMul Playground
This is a playground to test different `GEMM` implementations without having to deal with the complexity of `YAMI`.

The process is quite simple:
- define a `GEMM` function in C++, the function must have the same name as the containing file
- run `driver.py`, this will compile all the kernels (meaning all the files inside `kernels/`) as shared objects,
and then it will generate the ctypes bindings for all the kernels. Finally, the driver will time all the kernels
for different matmul sizes and plot them.
