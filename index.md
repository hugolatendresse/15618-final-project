Authors: [Hugo Latendresse](https://github.com/hugolatendresse) and [Matthew Katz](https://github.com/mhk197)

Jump to:
- [Summary](https://hugolatendresse.github.io/15618-final-project/#summary)
- [Background](https://hugolatendresse.github.io/15618-final-project/#background)
- [The Challenge](https://hugolatendresse.github.io/15618-final-project/#the-challenge)
- [Resources](https://hugolatendresse.github.io/15618-final-project/#resources)
- [Goals and Deliverables](https://hugolatendresse.github.io/15618-final-project/#goals-and-deliverables)
- [Platform Choice](https://hugolatendresse.github.io/15618-final-project/#platform-choice)
- [Schedule](https://hugolatendresse.github.io/15618-final-project/#schedule)

[//]: # (External Links:)

[//]: # (- [Project Proposal &#40;TODO&#41;]&#40;docs/Project%20Proposal.pdf&#41;)


## Summary

We will add the capability of inferring with a Mixture of Experts Transformer model to the FlexFlow Serve framework. We will also profile its performance and benchmark its per-token latency against a standard, non-accelerated service setting.

## Background

Mixture of Experts (MoE) is a machine learning technique that has surged in popularity as a method to keep resource requirements and latency low for training and serving increasingly large generative DL models. MoE replaces a functional unit, or layer, of a model with multiple “expert” networks that are each significantly smaller than the original layer. Each expert is trained to specialize in processing a subset of the input space, so a given input need only be processed by its corresponding expert. A router or gate network is placed at the beginning of the layer and determines which expert to send the input to. In MoE, the inputs of the model are routed to a single expert via a gate or routing network, which is placed before the experts. Only a proper subset of the whole model is used for forward passes. This reduces the computational budget needed for inference as well as per-token generation latency. 

FlexFlow Serve is an open-source compiler and distributed system for highly optimized LLM serving. It utilizes standard forms of multi-GPU model parallelism, as well as speculative decoding to accelerate inference. The project is led by Prof Jia and his research team.

The framework does not yet support serving LLMs with MoE architectures. We propose selecting a baseline MoE-based LLM, implementing what is needed in FlexFlow Serve to support it, and benchmarking its inference performance (per-token latency, in particular) against a traditional, non-accelerated service setting. This will entail recreating the MoE model’s architecture and forward pass in FlexFlow, writing some CUDA kernels to parallelize the MoE router mechanism and MoE MLP layers, and evaluating performance.


## The Challenge

The biggest challenge that we anticipate is onboarding ourselves onto the FlexFlow project. The repository is quite large and nuanced. We do not yet know what components of the architecture we choose are already implemented, and where exactly our starting point will be. We also do not know whether we need to create and tune a small speculative model (SSM) to enable speculative decoding for inference, which is a core technique for decreasing the latency of models served with FlexFlow. To begin tackling these challenges, we will meet with a member of Prof Jia’s group who is familiar with FlexFlow.

A core challenge of the parallelization work we will do is figuring out how to efficiently parallelize the MoE MLP layer. Depending on the architecture we choose, we will need to parallelize the execution of a subset of expert networks. This may entail two “nested” aspects of the layer: intra-expert parallelism, and inter-expert parallelism. Our current interpretation is that intra-expert parallelism reduces down to parallelizing matrix multiplication. Inter-expert parallelism could be more complex – parallelizing distinct matrix multiplications. We will try to experiment with different techniques and choose which performs best.


## Resources

We will be working off of the FlexFlow codebase, which can be found here[2]. We will be writing a combination of CUDA kernels and C++ code.

The architecture we implement will depend on what compute resources we have available to us. The RTX 2080 chip used by the GHC machines may be insufficient to accomplish our tasks. 

For example, in order to have Mixtral 8x7B Instruct in half-precision, we would need access to a  (multi)GPU machine with 90GB of VRAM in total. We would need 45GB of VRAM to work with the model quantized to 8 bits.


## Goals and Deliverables

- PLAN TO ACHIEVE
  - Select a Hugging Face model as a baseline model
  - Finalize the selection of hardware for the project  
  - Write a CUDA kernel(s) implementing the key components of our baseline MoE model, namely the Switch Transformer blocks consisting of routers (gate functional units) and experts (FFNs)
  - Develop a strategy to parallelize Switch Transformer blocks
  - Complete the implementation of a full MoE transformer by incorporating our work with existing FlexFlow CUDA kernels for the traditional transformer parts of MoE models (self-attention, etc.)    
  - Write other CUDA and C++ code to make our baseline model work with FlexFlow (inference only).
  - Successfully serve the model in FlexFlow 
  - Benchmark per-token latency of our baseline model using FlexFlow vs per-token latency without using an accelerator. 
  - Create a poster explaining how FlexFlow works, describing the architecture of our chosen MoE model, and showing our process in parallelizing it.
- 
- HOPE TO ACHIEVE
  - Continuously iterate on our implementation to achieve good speedups 
  - Benchmark our implementation against other accelerators like vLLM and FasterTransformer 
  - Create a web interface for interacting with our implementation vs non-accelerated implementation

## Platform Choice

FlexFlow is implemented in C++ and CUDA. 

## Schedule
    
  - Write other CUDA and C++ code to make our baseline model work with FlexFlow (inference only).
  - Successfully serve the model in FlexFlow 
  - Benchmark per-token latency of our baseline model using FlexFlow vs per-token latency without using an accelerator. 
  - Create a poster explaining how FlexFlow works, describing the architecture of our chosen MoE model, and showing our process in parallelizing it.

| Week              | Task                                                                                          | 
|-------------------|-----------------------------------------------------------------------------------------------|
| Nov. 11 - Nov. 17 | Meet with Gabriele or Zhihao and finalize the project proposal                                | 
| Nov. 11 - Nov. 17 | Select a Hugging Face model as a baseline model                                               | 
| Nov. 11 - Nov. 17 | Finalize the selection of hardware for the project                                            | 
| Nov. 18 - Nov. 24 | Write a CUDA kernel(s) implementing the Switch Transformer block                              | 
| Nov. 18 - Nov. 24 | Develop a strategy to parallelize Switch Transformer blocks                                   | 
| Nov. 18 - Nov. 24 | Complete the implementation of a full MoE transformer                                         | 
| Nov. 18 - Nov. 24 | Write other CUDA and C++ code to make our baseline model work with FlexFlow (inference only). | 
| Nov. 25 - Dec. 1  | Successfully serve the model in FlexFlow                                                      | 
| Dec. 2 - Dec. 8   | Benchmark our implementation with regular inference                                           | 
| Dec. 2 - Dec. 8   | Complete poster                                                                               | 
| Dec. 9 - Dec. 15  | Complete final report                                                                         | 

## References

[1] https://arxiv.org/abs/2305.09781
[2] https://github.com/flexflow/FlexFlow
[3] https://flexflow.readthedocs.io/en/latest/
