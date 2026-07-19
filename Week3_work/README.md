# Week 3 - Decoder-Only Transformer

Week 3 was the main architecture jump of the project. After seeing how LSTMs process sequences step by step, I built a GPT-style decoder-only Transformer in PyTorch to understand why attention became the core of modern language models.

## Goal

Implement and train a compact decoder-only Transformer for next-token prediction.

## Files

| File | Purpose |
| --- | --- |
| `transformer_model.py` | Main PyTorch Transformer implementation |
| `Copy of model_2.ipynb` | Notebook version of the experiment |
| `input.txt` | Text corpus used for tokenization and training |
| `../Mid-term_Report/assets/week3_architecture_flow.jpg` | Architecture diagram used in the report |
| `../Mid-term_Report/assets/week3_transformer_loss.jpg` | Saved train/validation loss plot |

## What Was Implemented

- GPT-2 tokenization through `tiktoken`.
- Sequential train/validation split of the text data.
- Decoder-only Transformer blocks.
- Causal self-attention so the model only attends to past tokens.
- Grouped-query attention with fewer key-value heads than query heads.
- Rotary positional embeddings for position-aware attention.
- RMSNorm normalization.
- SwiGLU feed-forward layers.
- AdamW optimization with cross-entropy loss.
- A `generate` method for sampling new tokens.
- Model checkpoint saving with `torch.save`.

## Result

The saved loss plot shows train and validation cross-entropy falling from about `10.9` to about `6.7` over the logged checkpoints. The final model generated sample text from a blank starting token.

This was still a small training run, but architecturally it connected directly to GPT-style models.

## How To Run

From the repository root:

```bash
python Week3_work/transformer_model.py
```

A CUDA GPU is recommended. The script trains for several thousand iterations, so it can take a while.

## Requirements

Relevant Python packages:

- `torch`
- `tiktoken`
- `matplotlib`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA calls the Transformer the cornerstone week, and that felt accurate. The central learning was scaled dot-product attention: queries, keys, and values let every token decide which earlier tokens are relevant. Multi-head attention then lets the model learn several kinds of relationships in parallel.

The other pieces are just as important as attention itself. Positional information tells the model where tokens are, residual connections preserve signal flow, normalization keeps training stable, and feed-forward layers give each token more computation after communication. Building these parts directly made the GPT-style architecture much easier to reason about.

## Takeaway

The Transformer made the biggest conceptual difference so far. Instead of squeezing a whole history into one hidden state, attention lets every token directly compare itself with earlier tokens. That one idea explains a lot of the power of modern LLMs.
