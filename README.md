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
- ~~RoPE~~
- ~~SwiGLU~~
- ~~LLaMA 2~~
- >Optimizations to reach GGML level
- >Quantization
- >GEMM/GEVM parallel
- Utility script to convert a PyTorch model from CLI
- Move tokenizer and weights to the same file

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
- ~~Switch to a multiple arenas scheme where each arena has its own lifetime~~
- ~~Add a `contiguous` flag~~
- ~~Use a decent GEMM, it's the main point of all this...~~
- ~~Remove `pthread`, use `OpenMP`~~
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
- Perf

## LLaMA 2
### TODO:
- Perf:
    - **90%** of the cycles are spent on `yami_gemm` and `yami_gevm` with GEMM taking up for almost all the cache-misses
      (`yami_packB`). This means that a bigger model did not introduce anything extra in terms of complexity but rather
      more operations to perform.

## Performance Roadmap
Main topic:
- Efficient parallelization with `OpenMP` or `pthread`
- `GEMM` optimization

Right now, I'd like to experiment a bit further with parallelization using `OpenMP` (and comparing it with `pthread`)
before diving into quantization and more esoteric stuff.

Another important aspect is to better define the framework we want to use to test performance.

## Comparison w/ GGML
Tested on my Ryzen 9 3900X w/ 64GB RAM running Linux `6.6.26-1-MANJARO` using `llama.cpp` commit `dd46dbc7`:

|     Model     |    Format     |      GGML       |   YAMI    |    PyTorch    |
|:-------------:|:-------------:|:---------------:|:---------:|:-------------:|
|  GPT-2 Small  |     FP32      |       ...       |    ...    |      ...      |  
|   LLaMA2 7B   |     FP32      |    1.1 tok/s    | 0.7 tok/s |   0.5 tok/s   |

