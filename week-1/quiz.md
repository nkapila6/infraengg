Answers are at the bottom of the page. Attempt all the questions first and then check how many you got correct.

**Q1.** During which phase of LLM inference is the KV cache populated for the prompt tokens?
- a) Decode - it grows one token at a time
- b) Prefill - K/V for all prompt tokens are computed and stored in one parallel pass
- c) Tokenization - the cache is built on the CPU host
- d) It is loaded from disk with the model weights

**Q2.** Which statement best describes why decode is memory-bandwidth bound?
- a) The decode matmuls are numerically unstable and must run in FP32
- b) Each decode step performs huge amounts of arithmetic on very little data
- c) Each decode step must read all model weights plus the growing KV cache from HBM to produce just one token's worth of math
- d) The KV cache is stored in CPU RAM and must cross PCIe every step

**Q3.** TTFT (time to first token) equals:
- a) Prefill time only
- b) Prefill time + first decode step
- c) Average decode step time × sequence length
- d) Tokenization time + network latency

**Q4.** In the lab, you ran a short prompt and a long prompt (the same text repeated 9×) through the same model. What happened to the timings?
- a) Prefill and decode both slowed proportionally
- b) Prefill slowed several-fold; decode tokens/sec stayed roughly flat
- c) Decode slowed several-fold; prefill stayed roughly flat
- d) Neither changed - the KV cache absorbs prompt length

**Q5.** Chunked prefill (as used by modern engines) primarily solves which problem?
- a) The KV cache growing without bound in long conversations
- b) Long prompts monopolizing the GPU and stalling other users' decode streams
- c) Weights not fitting in a single GPU's HBM
- d) Tokenizer overhead on the CPU host

**Q6.** Prefill/decode disaggregation means:
- a) Running prefill in FP8 and decode in FP16
- b) Skipping prefill for cached prefixes
- c) Running prefill on compute-optimized nodes and decode on memory-heavy nodes, shipping the KV cache between them
- d) Interleaving prefill chunks with decode steps on one GPU

**Q7.** Three things live in GPU VRAM during inference: weights, activations, KV cache. Which grows over time as generation proceeds?
- a) Weights
- b) Activations
- c) KV cache
- d) All three grow together

**Q8.** The KV cache is not a problem during prefill because:
- a) Prefill doesn't create any K/V tensors
- b) The cache is created once in a single parallel pass, whereas in decode it accumulates and persists step after step
- c) The cache is stored in SRAM during prefill
- d) Prefill compresses the cache with quantization automatically

**Q9.** Llama 3 8B uses GQA with 8 KV heads shared by 32 query heads. Compared to standard MHA, the KV cache per token is:
- a) 4× smaller - allowing ~4× more concurrent requests per HBM budget
- b) 4× larger - GQA trades memory for quality
- c) The same size - GQA only saves compute
- d) 32× smaller - one head instead of 32

**Q10.** GQA was adopted by Llama-class models primarily for:
- a) Compute savings during training
- b) Memory savings that translate directly into serving throughput
- c) Better model quality on long contexts
- d) Compatibility with FlashAttention

**Q11.** PagedAttention (vLLM) solves KV memory fragmentation by:
- a) Quantizing the cache to INT4
- b) Evicting the oldest tokens in a sliding window
- c) Storing the cache in fixed-size, non-contiguous blocks mapped by a block table - like OS virtual memory
- d) Moving the cache to CPU RAM between decode steps

**Q12.** In the lab's optional section, decoding WITHOUT the KV cache was dramatically slower because:
- a) Every step re-forwards the entire sequence instead of just the new token
- b) The profiler adds overhead when the cache is off
- c) use_cache=False forces FP32 computation
- d) The tokenizer re-runs on every step

**Q13.** You're deploying a chatbot with long conversations and high concurrency (a decode-heavy workload). Which spec-sheet row should you read FIRST?
- a) FP8 Tensor Core TFLOPS
- b) Memory bandwidth (then memory capacity)
- c) Max TDP
- d) PCIe generation

**Q14.** The H100 SXM offers 3.35 TB/s (HBM3) vs the L40S at 0.864 TB/s (GDDR6). For single-stream decode of the same model, roughly what speedup should the H100 give?
- a) None - decode depends on TFLOPS, and both have Tensor Cores
- b) ~2×
- c) ~4× - decode tok/s scales with memory bandwidth
- d) ~40× - bandwidth compounds per layer

## Answers

**Q1 - Answer: b**

Prefill computes Q, K, V for every prompt token in one parallel pass and stores K/V - that IS cache population. Decode then appends to it one token at a time.

**Q2 - Answer: c**

Per decode step the arithmetic is tiny (one token's forward pass ≈ 2 × params FLOPs) but the reads are huge: all weights + the whole cache stream from HBM. The GPU waits on bytes, not math.

**Q3 - Answer: b**

TTFT = prefill + first decode step. Streaming speed thereafter is decode throughput (tok/s). Two metrics, two phases, two bottlenecks.

**Q4 - Answer: b**

Prefill scales with prompt length (more tokens in the parallel pass). Each decode step processes one token regardless of prompt length, so tok/s barely moves (the slightly larger cache adds only a small read cost).

**Q5 - Answer: b**

Chunked prefill splits long prompts into ~512-2048-token chunks interleaved with other requests' decode steps - no user's stream freezes because someone pasted a novel. (a) is addressed by paging/quantization/windows; (c) by parallelism.

**Q6 - Answer: c**

At large scale, the two phases get physically separate hardware pools matched to their bottlenecks (compute-optimized vs memory-heavy), with the KV cache transferred between clusters. (d) describes chunked prefill instead.

**Q7 - Answer: c**

Weights are fixed; activations are transient; only the KV cache accumulates - with every generated token, for every request in flight.

**Q8 - Answer: b**

Creation is a one-shot, amortized, parallel cost. Accumulation is the problem: decode appends K/V per layer per head every step, and the cache persists for the life of the request.

**Q9 - Answer: a**

KV per token scales with the number of KV heads: 8 instead of 32 → 4× smaller cache → 4× more concurrent tokens per HBM budget.

**Q10 - Answer: b**

The adoption argument was memory, not compute: a smaller cache per token converts directly into more concurrent requests (throughput) on the same hardware.

**Q11 - Answer: c**

Naive serving pre-allocates contiguous max-length cache slabs, wasting 60-80% of KV memory (per the vLLM paper). Paged blocks + a block table kill fragmentation and enable prefix sharing.

**Q12 - Answer: a**

Without past_key_values, step N re-processes all prompt + N generated tokens. Work grows quadratically overall - the lab makes this painfully visible in ten steps.

**Q13 - Answer: b**

Decode-heavy → bandwidth governs tok/s; capacity governs how many concurrent conversations' caches fit. TFLOPS would be the first read for a prefill-heavy batch workload.

**Q14 - Answer: c**

Decode tok/s ≈ bandwidth ÷ bytes per token. 3.35 ÷ 0.864 ≈ 3.9 - call it ~4×, straight off the datasheet.
