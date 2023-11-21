# YAMI
Yet Another Machine Inference framework

## TODO
**YAMI2:**
- ~~matmul up to 4 dim~~
- ~~add, addv, mul, mulv, div, divv~~
- ~~view (reshape)~~
- ~~transpose~~
- ~~mask~~
- ~~tanh~~
- ~~softmax~~
- ~~GELU~~
- >**GPT-2!**
    - Pytorch, on my machine, takes something like 75-80ms to generate a single token
using multiple threads (*to be verified*). This is the number to beat
- ~~In-place operations~~
- Add the possibility to allocate a temporary `yami_tensor`. The easiest and probably the best
way to do so is to treat the `yami_context` as a sort of stack where you can just call `pop`
to "free" the last tensor


**Extra:**
- ~~BPE (C++)~~
