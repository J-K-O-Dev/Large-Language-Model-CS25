# Week 7 - QLoRA-Style Efficient Tuning

Week 7 explored how to adapt a large model without fully fine-tuning every parameter. This is where quantization and LoRA become practical tools instead of just research-paper ideas.

The notebook loads Llama 3 8B in 4-bit precision, prepares it for low-bit training, injects LoRA adapters, and trains only those adapter weights with a DPO-style preference setup.

## Goal

Use 4-bit quantization and LoRA adapters to make an 8B-parameter model trainable on limited hardware.

## Files

| File | Purpose |
| --- | --- |
| `Quantization_technique.ipynb` | 4-bit Llama 3 loading, PEFT LoRA setup, and DPO training |

## What Was Implemented

- Loaded `meta-llama/Meta-Llama-3-8B-Instruct`.
- Used `BitsAndBytesConfig` with:
  - `load_in_4bit=True`
  - `bnb_4bit_quant_type="nf4"`
  - `bnb_4bit_use_double_quant=True`
  - `bnb_4bit_compute_dtype=torch.bfloat16`
- Prepared the model for k-bit training with PEFT.
- Added LoRA adapters with:
  - rank `r=16`
  - `lora_alpha=32`
  - dropout `0.05`
  - target modules including attention and MLP projections
- Used `Anthropic/hh-rlhf` as the preference dataset.
- Configured `DPOTrainer` on a small subset with `beta=0.3`.
- Used `paged_adamw_8bit` to reduce optimizer memory pressure.

## Result

The PEFT setup dramatically reduced the number of trainable parameters:

| Metric | Value |
| --- | ---: |
| Trainable parameters | `41,943,040` |
| Total parameters | `8,072,204,288` |
| Trainable percentage | `0.5196%` |

The recorded training summary showed:

| Metric | Value |
| --- | ---: |
| Global step | `300` |
| Training loss | `0.635` |
| Epochs | `3.0` |
| Runtime | about `3296` seconds |

## How To Run

Open the notebook:

```bash
jupyter notebook Week7_work/Quantization_technique.ipynb
```

This notebook is best run in Colab or a Linux CUDA environment. It also requires Hugging Face access to the gated Llama 3 model.

## Requirements

Relevant Python packages:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `bitsandbytes`
- `peft`
- `trl`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA frames this week around deployable and trainable LLMs under real hardware limits. Full fine-tuning updates every parameter, which is usually impossible for 7B+ models on small GPUs. LoRA solves this by adding small trainable low-rank matrices while keeping the base model frozen.

QLoRA goes one step further by keeping the base model in 4-bit precision and training only the adapters. That is why the trainable-parameter percentage matters so much here: the experiment trained a tiny fraction of the model while still adapting the behavior.

## Takeaway

This week made efficient fine-tuning feel practical. Instead of updating all 8B parameters, the run trained about half a percent of the model through adapters while the base stayed quantized and frozen.
