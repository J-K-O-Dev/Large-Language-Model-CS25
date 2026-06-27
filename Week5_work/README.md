### Week 5 Report: Efficient Fine-Tuning with LoRA

Week 4 was a great introduction to the power of adapting large, pretrained models. However, it also highlighted a major bottleneck: full fine-tuning, where every weight in the model is updated, is incredibly resource-intensive. It requires a huge amount of VRAM and results in a full-sized model checkpoint for every new task. This week, I tackled that problem head-on by exploring **Parameter-Efficient Fine-Tuning (PEFT)**, specifically a popular and powerful technique called **Low-Rank Adaptation (LoRA)**.

#### The Theory: Training a Fraction to Tune the Whole

The core idea behind PEFT is to adapt a large language model to a new task without retraining all of its parameters. LoRA is a particularly clever way to do this.

- **The Problem with Full Fine-Tuning:** A model like GPT-2 has over 100 million parameters. When you fine-tune it, you're calculating gradients and updating every single one of them. This is slow, memory-hungry, and means that if you fine-tune the model for 10 different tasks, you end up with 10 separate, large model copies.

- **The LoRA Solution:** LoRA hypothesizes that the _change_ in the weights needed for fine-tuning has a low "intrinsic rank." This means the update can be represented efficiently. Instead of updating the massive original weight matrix `W`, LoRA freezes it and injects a pair of small, trainable "adapter" matrices (`A` and `B`) into the model's layers (typically the attention layers).
  - The update is decomposed into two low-rank matrices: `ΔW = B * A`.
  - During training, only `A` and `B` are updated. Since their rank `r` is much smaller than the original dimensions, the number of trainable parameters is drastically reduced (often by over 99%).
  - For inference, the learned weights can be merged back into the original weights (`W' = W + BA`) with no additional latency.

This approach makes fine-tuning much more accessible, allowing for faster training on consumer hardware and creating tiny, portable adapter checkpoints (often just a few megabytes) for each task.

#### Implementation: LoRA for Code Generation with PEFT

To test this, I created the `LoRA_Fine_tuning.ipynb` notebook. I decided to use the same GPT-2 model and MBPP dataset from Week 4 to get a direct comparison against full fine-tuning. The implementation relied heavily on the Hugging Face `peft` library.

- **Setting up LoRA:** The process was surprisingly straightforward:
  1.  Load the pretrained `gpt2` model and tokenizer.
  2.  Define a `LoraConfig` object. Here, I specified the `r` (rank), `lora_alpha` (a scaling factor), `target_modules` (telling the library which layers to adapt, in this case, the attention layers `c_attn`), and `lora_dropout`.
  3.  Use the `get_peft_model` function to wrap the base model with the LoRA adapters.

- **Drastic Parameter Reduction:** The most striking part was seeing the numbers. The wrapped model had only a tiny fraction of its parameters set as trainable. This immediately demonstrated the efficiency of the method—I was preparing to train less than 1% of the model's total weights.

- **Training with the `Trainer` API:** The beauty of the `peft` library is its seamless integration with the existing Hugging Face ecosystem. The same `Trainer` and `TrainingArguments` workflow from Week 4 worked with the PEFT model without any major changes. The library handles the logic of freezing the base model and only updating the adapter weights during backpropagation.

- **Saving the Adapter:** After training, instead of saving another 500MB+ model, I only had to save the learned adapter weights. The resulting checkpoint was just a few megabytes, making it incredibly easy to store and share.

#### Key Takeaways

This week was a huge "aha!" moment regarding the practicality of LLMs. While building a Transformer in Week 3 was about understanding the architecture, and full fine-tuning in Week 4 was about the standard workflow, this week was about making it all _feasible_.

The core learning was that you don't need to move a mountain to change its path. LoRA showed that by training a very small number of well-placed parameters, you can steer a massive pretrained model toward a new task with surprisingly good results. The performance on the code generation task was comparable to the fully fine-tuned model from last week, but the training was faster and used significantly less memory.

This feels like a critical piece of the modern LLM puzzle. Techniques like LoRA are what enable developers and researchers without access to massive GPU clusters to experiment with and deploy powerful models. It's the bridge between the theoretical power of LLMs and their real-world application. I'm now much more confident that I can take on even larger models, knowing there are efficient ways to adapt them.
