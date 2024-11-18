# YAMI
Yet Another Machine Inference framework.

## Goal
YAMI is a C++ library inspired by [ggml](https://github.com/ggerganov/ggml) to run Large Language Models (LLMs)
locally on CPU.

The goal is not to build the best inference engine on the market but rather to learn about Large Language Models
and optimisation topics in a way that is fun (at least for me!).
For this reason no external library is used.

The thing I'm most proud of is the GEMM kernel I developed for YAMI following the famous
[Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
paper. Specifically, the kernel I developed on a Linux system running an AMD Ryzen 9 3900X CPU (AVX2) achieves
similar performance as [OpenBLAS](https://www.openblas.net/).

## Requirements
The requirements for YAMI are:
- A modern C++ compiler with good support for C++17
- GNU Make
- Python >= 3.9

For the weights conversion script you'll have to install the requirements in the `requirements.txt` file:
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Usage
To use YAMI first build the model you want to inference, for example GPT-2:
```bash
make gpt2 YAMI_FAST=1
```
Next, if you haven't already done so, create a YAMI-compatible file with the weights of your model.
There is a convenience script `convert_yami.py` to do so:
```bash
./convert_yami.py models/gpt2 -v
```
This will create a file like `yami_gpt2_fp32.bin` in the `models/gpt2` folder.

Finally, to run the model simply do:
```bash
./gpt2 -i "<Input prompt here>" -m models/gpt2/yami_gpt2_fp32.bin
```

To check all the options for each model use the `--help` flag.

### Notes on profiling
To profile the kernels one can specify the `YAMI_TRACE=1` flag at compile time.
This will instrument YAMI to take measurements of the execution time (as well as the total GFLOPS)
of the most computationally intensive kernels such as GEMM, GEVM ecc...

**Note:** doing so will cause a print to be issued after each layer of your model. This will for sure flood your stdout!

## Comparison w/ GGML
Tested on my Ryzen 9 3900X w/ 64GB RAM running Linux `6.6.26-1-MANJARO` using `llama.cpp` commit `dd46dbc7`:

|     Model     |    Format     |      GGML       |   YAMI    |    PyTorch    |
|:-------------:|:-------------:|:---------------:|:---------:|:-------------:|
|  GPT-2 Small  |     FP32      |       ...       |    ...    |      ...      |  
|   LLaMA2 7B   |     FP32      |    1.1 tok/s    | 0.7 tok/s |   0.5 tok/s   |

## License
MIT