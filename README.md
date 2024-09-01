### Course Itinerary

## Session 1
**Introduction: How Libraries Work**

- Motivation for the course and understanding computational efficiency.
- Factors affecting the efficiency of model performance.

**Seminar:**
- Measuring time, memory, and how autograd works.

## Session 2
**Hardware and Low-Level Solutions**

- Introduction to computational devices, how CPU and GPU memory work.

**Seminar:**
- Profiling models with Pytorch Profiler.

## Session 3
**Automatic Architecture Search**

- Methods of automatic architecture search, including computationally efficient models.

**Seminar:**
- Differentiated architecture search and evolution.

## Session 4
**JIT and Compile in Pytorch 2.0**

- Deep dive into JIT trace, jit compile, and Compile.

**Seminar:**
- Working with JIT, converting models to ONNX, converting models to TensorRT. Speeding up models with JIT and compile.

## Session 5
**Pruning and Sparsification**

- Overview of main methods of model sparsification, motivation for why it works, and types of sparsification. Methods of sparsification for LLM.

**Seminar:**
- Structured and unstructured pruning for VGG, iterative pruning, and magnitude-based pruning.

## Session 6
**Seminar:**
- Hyperparameter tuning for optimal model sparsification within a given budget.

**Seminar:**
- Using the low-level library Triton.

## Session 7
**Quantization**

- Main methods and approaches to quantization, overview of LLM quantization methods.

**Seminar:**
- Implementing quantization with LSQ. Quantization with Pytorch, quantization with ONNX.

## Session 8
**Optimization Methods for Large Language Models (LLM)**

- KV-Cache, Paged Attention, GradientCheckpointing, and more.

**Seminar:**
- Using QLora - a library.

## Session 9
**Tensor Decompositions for Language Models**

- Main methods of tensor decomposition for language models. What can be achieved with TD and when is it best to use?
- Introduction to TD. Description of general methods and concepts. Overview of modern methods of model optimization using TD. Overview of libraries.

**Seminar:**
- Instead of fully connected layers, use their compressed representation obtained with SVD.
