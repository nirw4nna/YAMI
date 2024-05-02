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
- >LLaMA 2

**Extra:**
- ~~BPE (C++)~~
- ~~LLaMA tokenizer (C++)~~

## Refactoring
There are few things I dislike about the current state of `YAMI`.
First of, let's clarify the scope of this project: this is not ment to become the next `ggml` nor the next
[insert name here] best framework to do inference of LLMs.
`YAMI` was born with one simple objective: to become a playground for me to learn more about software optimizations.

- The goal is to provide a fully self-contained, high performance implementation of popular LLMs on x86.
At least until we get there, it doesn't make sense to waste time thinking about CUDA, Metal, ARM, Rocm or whatever.

- Same goes for the operating system: it doesn't make sense to worry about Windows, macOS or any other OS until there will be
a good reason to do so

With this out of the way, let's try and fix some things in order to make this thing worth publishing:
- ~~Move to `u64` for sizes~~
- Verify the usefulness (profiling) of in-place operations, with `__restrict` and some vectorization we could probably
gain significant performance boost
- ~~Remove the `extended_dim` abstraction, it's useless~~ 
- Switch to a multiple arenas scheme where each arena has its own lifetime
- ~~Add a `contiguous` flag~~
- Use a decent GEMM, it's the main point of all this...
- Remove `pthread`, use `OpenMP`
- _`YAMI_MAX_DIMS` should be defined at compile-time (this is the trickiest of them all,
it has to do with recursive macros like `yami_for` and `yami_offset` and it's also used indirectly in functions
that operate along axis like `yami_sum`. I would not bother with this for the time being)._
- I don't like the way scalar/vector types are handled in functions like `yami_div`
- I don't like the way the indexes are computed in functions like `yami_sum`

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
