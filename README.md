# YAMI
Yet Another Machine Inference framework

## Open topics
**Core:**
- ~~matmul up to 4 dim~~
- ~~add, addv, mul, mulv, div, divv~~
- ~~view (reshape)~~
- ~~transpose~~
- ~~mask~~
- ~~tanh~~
- ~~softmax~~
- ~~GELU~~
- ~~GPT-2~~
- ~~RMSNorm~~
- >RoPE
- ~~SwiGLU~~
- >Optimizations to reach GGML level
- Tensor allocator with support for "temporary" allocations
- CUDA support
- >LLaMA 2



**Extra:**
- ~~BPE (C++)~~
- ~~LLaMA tokenizer (C++)~~

## GPT2
### Benchmarks:
- Pytorch, on my machine, takes something like 75-80 ms to generate a single token
using multiple threads.
- GGML takes something around 35-40 ms to generate a single token without doing anything extra fancy, this is very good
and also means we can do much better
- Vanilla NumPy takes something around 120-200 ms to generate a single token

### TODO:
- There are a couple of things that can be added to `yami_internal_matmul_blocked` to, hopefully,
speed up the calculations (more care when packing B and better testing with block factor).
Also, start experimenting with **SIMD** instead of relying solely on the compiler.
- Thorough profiling using `perf`

## LLaMA 2
