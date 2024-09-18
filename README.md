
# PART 1: HARDWARE AND LOW-LEVEL OPTIMIZATION

## Session 1
### Introduction: How Libraries Work

- **Motivation for the course and understanding computational efficiency.**
- **Factors affecting the efficiency of model performance.**

**Seminar:**
- Measuring time, memory, and how autograd works.

## Session 2
### Hardware and Low-Level Solutions

- **Introduction to computational devices, how CPU and GPU memory work.**

**Seminar:**
- Profiling models with Pytorch Profiler.

## Session 3
### Automatic Low-Level Optimization

**Seminar:**
- Working with JIT, converting models to ONNX, converting models to TensorRT. Speeding up models with JIT and compile.

## Session 4
### Quantization

- **Main methods and approaches to quantization, overview of LLM quantization methods.**

**Seminar:**
- Implementing quantization with LSQ. Quantization with Pytorch, quantization with ONNX.

## Session 5
### Pruning and Sparsification

- **Overview of main methods of model sparsification, motivation for why it works, and types of sparsification. Methods of sparsification for LLM.**

**Seminar:**
- Structured and unstructured pruning for VGG, iterative pruning, and magnitude-based pruning.

# PART 2: OPTIMIZING LLMs

## Session 6
### Low-Level and Algorithmic Optimization Methods for Large Language Models (LLM)

**Seminar:**
- Fine-tuning of Quantized LLM

## Session 7
### Tensor Factorization for LM

- **Main methods of tensor decomposition for language models. What can be achieved with TD and when is it best to use?**
- **Introduction to TD. Description of general methods and concepts. Overview of modern methods of model optimization using TD. Overview of libraries.**

**Seminar:**
- Instead of fully connected layers, use their compressed representation obtained with SVD.

# PART 3: IN SEARCH OF GOOD MODELS

## Session 8
### Automatic Architecture Search

- **Methods of automatic architecture search, including computationally efficient models.**

**Seminar:**
- Differentiated architecture search and evolution.
