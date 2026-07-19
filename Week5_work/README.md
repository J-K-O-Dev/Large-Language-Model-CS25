# Week 5 - Evaluation, Scaling, And Model Loading

Week 5 focused on two practical questions: how do we compare language models, and what happens when model size starts running into hardware limits?

The main quantitative experiment was a perplexity comparison across GPT-2 model sizes. The second notebook explored loading an instruction/chat model under limited hardware, using TinyLlama as a practical substitute where a larger Llama checkpoint would be too heavy.

## Goal

Evaluate language-model scaling through perplexity and practice loading a chat model for inference.

## Files

| File | Purpose |
| --- | --- |
| `perplexity_comparision_gpt2.ipynb` | GPT-2 perplexity comparison on WikiText-2 |
| `gpt2_perplexity_scaling.png` | Saved perplexity-vs-scale plot |
| `Load_LLaMA3_8B (1).ipynb` | Model loading/inference experiment using TinyLlama because of hardware limits |

## What Was Implemented

- Downloaded the WikiText-2 test text for evaluation.
- Loaded four GPT-2 variants: `gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`.
- Used a sliding-window evaluation strategy with a stride of 512 tokens.
- Computed negative log-likelihood and converted it to perplexity.
- Plotted perplexity against parameter count on a log-scaled x-axis.
- Loaded `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with Hugging Face `transformers`.
- Tried model inference with automatic device placement and a Flash Attention path where available.

## Result

The GPT-2 scaling plot showed the expected pattern: larger models achieved lower perplexity on WikiText-2.

Approximate values from the saved plot:

| Model | Parameters | WikiText-2 perplexity |
| --- | ---: | ---: |
| `gpt2` | 124M | `19.6` |
| `gpt2-medium` | 355M | `15.3` |
| `gpt2-large` | 774M | `13.4` |
| `gpt2-xl` | 1.5B | `12.6` |

The TinyLlama notebook confirmed the practical side of model loading: the model and tokenizer can be loaded through Hugging Face and moved onto available hardware with `device_map="auto"`.

## How To Run

Open the notebooks:

```bash
jupyter notebook Week5_work/perplexity_comparision_gpt2.ipynb
jupyter notebook "Week5_work/Load_LLaMA3_8B (1).ipynb"
```

The perplexity notebook downloads models and the WikiText-2 test text, so it needs internet access. Running all GPT-2 variants is much easier with a CUDA GPU.

## Requirements

Relevant Python packages:

- `torch`
- `transformers`
- `matplotlib`
- `numpy`
- `tqdm`
- `accelerate`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA uses this week to connect model quality with scale. Kaplan-style scaling laws say performance improves predictably with more parameters, data, and compute, while Chinchilla-style results remind us that the balance between model size and tokens matters just as much.

The GPT-2 perplexity experiment made that idea measurable: as the parameter count increased, the model became less surprised by the same WikiText-2 text. The TinyLlama loading experiment added the hardware lesson: modern LLM work is also about memory, precision, attention kernels, and choosing a model that can actually run on available hardware.

## Takeaway

Perplexity made scaling laws visible. Bigger models were not just bigger for the sake of it; they were measurably less surprised by the same evaluation text. At the same time, the model-loading notebook made the hardware wall very concrete.
