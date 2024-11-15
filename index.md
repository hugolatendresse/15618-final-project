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

We will add the capability of inferring with a Mixture of Experts model (a transformer with MoE, for MLP layers only) to the FlexFlow Serve framework. Once we are able to infer with an MoE model in FlexFlow, we will benchmark its per-token latency and throughput on a node with four V100 GPUs against a standard, non-accelerated service setting.

## Background

Mixture of Experts (MoE) is a machine learning technique that has surged in popularity as a method to keep resource 
requirements and latency low for training and serving increasingly large generative DL models. This technique is most 
commonly used for LLMs, which our project will focus on. MoE replaces a functional unit, or layer, of a model with 
multiple “expert” networks that are each significantly smaller than the original layer. Each expert is trained to 
specialize in processing a subset of the input space, so a given input need only be processed by its corresponding 
expert. A router or gate network is placed at the beginning of the layer and determines which expert to send the 
input to. Only a proper subset of the whole model is used for forward passes. This reduces the computational needs 
and can help improve the per-token generation latency. 

MoE is most commonly applied to MLP layers of transformers, so that is what we will focus on. As most layers of modern LLMs, those layers require a lot of computation and memory access. Moreover, modern LLMs do not fit on a single GPU. Therefore, parallelization is necessary to serve them. The question is not "whether" to parallelize, but "how" to parallelize, with the goal of achieving the best possible latency and throughput on a given set of hardware resource. Different parallelization strategies will lead to different data movement, amount of redundant computation, etc., leading to different performance. 

FlexFlow Serve is a project led by Prof Jia and his research team. It is an open-source compiler and distributed system for highly optimized LLM serving. It utilizes standard forms of multi-GPU parallelism, as well as speculative decoding to accelerate inference. For our research project, we will focus on incremental decoding: speculative decoding is not in scope. 

FlexFlow does not yet support serving LLMs with MoE architectures. We propose selecting a baseline MoE-based LLM, implementing what is needed in FlexFlow Serve to support it, and benchmarking its inference performance (per-token latency and throughput) against a traditional, non-accelerated service setting. This will entail recreating the MoE model’s architecture and forward pass in FlexFlow, writing some CUDA kernels to parallelize the MoE router mechanism and/or MoE MLP layers, and evaluating performance.




## The Challenge

The first challenge is to onboard ourselves onto the FlexFlow project. The repository is quite large and nuanced. We do 
not yet know what components of the architecture we choose are already implemented, and where exactly our starting point will be. 

The workload consists of an entire transformer model, though our focus will be on the MoE-MLP layers. Each layer of the 
model is of course dependent on the previous layer (and not vice-versa, since we are focusing on inference). Within an MoE-MLP layer,
the "experts" (FNN models) are dependent on the router. 

Temporal locality is high for the weights, but very limited for the activations. We expect spatial locality to be high within experts, but possibly lower across experts. 

The core challenge of the parallelization work we will do is figuring out the best way parallelize the MoE layer. Inferring 
with a model that cannot fit on a single GPU necessitates expensive communication, increasing the communication-to-computation ratio. 
There are infinitely many ways to decompose and schedule the calculation on multiple workers. Experimentation is needed to find the strategy that minimizes 
the amount of communication (and other costs). We will try to experiment with different techniques, including 
intra-expert parallelism and inter-expert parallelism (see goals section), and choose which one performs best. 

Divergent execution is typical to neural networks due to the non-linear activation functions. In MoE models, an additional 
layer of divergence is introduced due to the routing to experts. That can introduce load imbalance, as some experts may receive more inputs than others. 

TODO Describe constraints: What are the properties of the system that make mapping the
workload to it challenging?


## Resources

We will be working off of the FlexFlow codebase, which can be found in [1]. We will be writing a combination of CUDA kernels and C++ code.

We plan to use one node in Pittsburgh's Supercomputer (Bridges-2) GPU cluster. Each compute node has four V100 GPUs with 32GB of 
VRAM, for a total of 128GB for VRAM. We will tentatively base our project on Mixtral 8x7B Instruct with half-precision, which requires 90GB of VRAM in total. 


## Goals and Deliverables

TODO complete after reading https://www.cs.cmu.edu/afs/cs/academic/class/15418-f24/www/projects/project-proposal.pdf
(they say it's the most improtant section)

We plan on trying two main strategies to parallelize MoE layers: intra-expert parallelism, and inter-expert parallelism.
Intra-expert parallelism means processing the activations within a single expert concurrently. It is similar to the 
concept of data parallelism, but at a finer granularity (we split across activations within a sample instead of samples 
within a batch). 
Inter-expert parallelism means treating each expert or groups of experts as separate models that can operate independently
on different GPUs. It is more similar to model parallelism.  
We will tackle intra-parallelism first, and inter-parallelism second. 



- PLAN TO ACHIEVE
  - Write a CUDA kernel(s) implementing the key components of our baseline MoE model, namely a MoE MLP layer consisting of routers (gate functional units) and experts (FFNs)
  - Complete the implementation of a full MoE transformer by incorporating our work with existing FlexFlow CUDA kernels for the traditional parts of MoE transformers (self-attention, etc.)    
  - Write other CUDA and C++ code to make our baseline model work with the FlexFlow API (inference only).
  - Successfully serve an MoE model with FlexFlow.
  - Benchmark per-token latency and throughput of our baseline model using FlexFlow vs per-token latency of Hugging Face's transformers package. 
  - All steps above will be completed twice: once for intra-expert parallelism and once for inter-expert parallelism. 
  - Create a poster explaining how FlexFlow works, describing the architecture of our chosen MoE model, and showing our process in parallelizing it.


- HOPE TO ACHIEVE
  - Beat Hugging Face transformers in terms of per-token latency. We only "hope" to achieve that because that library already makes use of parallelization and is probably optimized to some extent, and because we are not using speculative decoding.     
  - Iterate on our implementation to achieve good speedups 
  - Benchmark our implementation against other accelerators like vLLM and FasterTransformer 
  - Create a web interface for interacting with our implementation vs non-accelerated implementation


- IF PROGRESS IS SLOW
  - In case the data parallelism takes longer than we thought, we may not complete the model parallelism. 

## Platform Choice

FlexFlow is implemented in C++ and CUDA, so those are the two languages we will use. 
We will be working from Linux machines since it is the only OS officially supported by FlexFlow. 

## Schedule

TODO complete after reading https://www.cs.cmu.edu/afs/cs/academic/class/15418-f24/www/projects/project-proposal.pdf


| Week                  | Task                                                                                                        | 
|-----------------------|-------------------------------------------------------------------------------------------------------------|
| Nov. 11 - Nov. 17     | Finalize the project proposal                                                                               | 
| Nov. 11 - Nov. 17     | Familiarize ourselves with the FlexFlow repo                                                                | 
| Nov. 11 - Nov. 17     | Go through FlexFlow's developer guide                                                                       | 
| Nov. 11 - Nov. 17     | Meet with a member of Prof Jia's reserach team                                                              | 
| Nov. 18 - Nov. 24     | Confirm the choice of baseline model, based on the resources available on PSC                               | 
| Nov. 18 - Nov. 24     | Develop a strategy to parallelize the baseline model with intra-expert parallelism                          | 
| Nov. 18 - Nov. 24     | Write a CUDA kernel(s) implementing an MoE MLP layer (with intra-expert parallelism)                        | 
| Nov. 18 - Nov. 24     | Complete the implementation of a full MoE transformer (with intra-expert parallelism)                       | 
| Nov. 18 - Nov. 24     | Write other CUDA and C++ code to make our baseline model work with FlexFlow (with intra-expert parallelism) | 
| Nov. 18 - Nov. 24     | Complete milestone report regarding with intra-expert parallelism                                           | 
| Nov. 18 - Nov. 24     | Start working on inter-expert parallelism (similar steps as for intra-expert parallelism)                   | 
| Nov. 25 - Dec. 1      | Complete inter-expert parallelism implementation and successfully serve the model in FlexFlow with it       | 
| Dec. 2 - Dec. 8       | Benchmark our implementation with regular inference                                                         | 
| Dec. 2 - Dec. 8       | Complete poster                                                                                             | 
| Dec. 9 - Dec. 15      | Complete final report                                                                                       | 

## References

TODO complete after reading https://www.cs.cmu.edu/afs/cs/academic/class/15418-f24/www/projects/project-proposal.pdf


[1] https://github.com/flexflow/FlexFlow  
