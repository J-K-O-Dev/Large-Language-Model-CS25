### Week 5 Report: Evaluating and Running Large-Scale Models

After fine-tuning models in Week 4, a natural next step was to understand how we measure if one model is "better" than another. This week, I moved into the world of model evaluation and the practical engineering challenges of running truly large-scale models. The work was twofold: first, I conducted a systematic evaluation of GPT-2 variants to observe **scaling laws** in action using perplexity. Second, I tackled the challenge of loading and running a much larger, state-of-the-art model, Llama 3 8B, which required exploring techniques like quantization.

#### The Theory: Scaling Laws, Perplexity, and Quantization

- **Perplexity (PPL):** A core metric for evaluating language models. It measures how well a probability model predicts a sample. In simple terms, it's a measure of "surprise." A lower perplexity means the model is less surprised by the test data, indicating it has learned the underlying language patterns more effectively.

- **Scaling Laws:** A key finding in LLM research is that model performance (like perplexity) improves predictably as you increase model size, dataset size, and computational budget. This experiment aimed to verify the model size aspect: as the number of parameters in a model increases, its perplexity on a standard benchmark should decrease.

- **Quantization:** Running a model with billions of parameters like Llama 3 (8B) is impossible on consumer GPUs, as the weights alone (in 16-bit precision) would require over 16GB of VRAM. **Quantization** is a technique to reduce this memory footprint. It involves converting the model's weights from high-precision floating-point numbers (like FP16) to lower-precision integers (like 8-bit or 4-bit). This drastically reduces the model size and memory usage, often with only a small impact on performance, making large models accessible.

#### Implementation Part 1: Verifying Scaling Laws with GPT-2

The work in `perplexity_comparision_gpt2.ipynb` was a direct test of scaling laws.

- **Models:** I evaluated four standard GPT-2 variants: `gpt2` (124M), `gpt2-medium` (355M), `gpt2-large` (774M), and `gpt2-xl` (1.5B).
- **Evaluation:** I calculated the perplexity for each model on the test set of the WikiText-2 dataset. To handle sequences longer than the model's context window (1024 tokens), I used a sliding window approach with a stride of 512 tokens. This ensures the entire dataset is evaluated without losing context.
- **Results:** The results clearly demonstrated the scaling law. As the model size increased from 124M to 1.5B parameters, the perplexity consistently dropped. This was a powerful, practical demonstration of why bigger models have become the standard.

#### Implementation Part 2: Running Llama 3 8B with Quantization

The second part of this week's work, documented in `load_llama3_8b.ipynb`, was to run a model far too large for my hardware.

- **The Challenge:** The Llama 3 8B model is a massive leap from GPT-2. Loading it directly would cause an out-of-memory error on a typical Colab GPU.
- **The Solution:** I used the `bitsandbytes` library integrated with Hugging Face `transformers` to load the model in 4-bit precision (`bnb_4bit_quant_type="nf4"`). This involves loading the base model with a `BitsAndBytesConfig` that tells the library to quantize the weights on the fly.
- **Inference:** After successfully loading the quantized model, I ran a few sample prompts through its generation pipeline. While not a rigorous evaluation, it confirmed that the model was functional and could produce coherent text even after its weights were compressed into a 4-bit format.

#### Key Takeaways

This week connected the dots between model architecture, evaluation, and deployment. It's one thing to train a model; it's another to measure its performance and figure out how to actually run it.

- **Scaling Laws are Real:** Seeing perplexity drop predictably as parameter count increased was a concrete validation of one of the most fundamental principles driving LLM development.
- **Perplexity as a Benchmark:** This week gave me hands-on experience with perplexity as a core evaluation metric. It's a standardized way to compare different language models on their fundamental ability to model language.
- **The Memory Wall is Real:** My attempt to load Llama 3 8B without quantization failed instantly. This made the hardware and memory constraints of LLMs incredibly tangible. It's not just about theory; it's about engineering.
- **Quantization is a Key Enabler:** Successfully running an 8B parameter model on a free Colab instance felt like a superpower. It showed that techniques like 4-bit quantization are not just academic tricks but essential tools for making LLMs practical for a wider audience.
