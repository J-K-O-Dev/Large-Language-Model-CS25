# Week 6 - Direct Preference Optimization

Week 6 moved from supervised fine-tuning into preference alignment. The question this week was: once a model can generate text, how do we push it toward responses that humans prefer?

I explored Direct Preference Optimization, or DPO, which is a simpler alternative to full RLHF. Instead of training a separate reward model and running reinforcement learning, DPO directly optimizes the model on chosen/rejected response pairs.

## Goal

Use DPO to align a small chat model with human preference data.

## Files

| File | Purpose |
| --- | --- |
| `DPO_Fine_Tuning.ipynb` | Main DPO training notebook |

## What Was Implemented

- Loaded `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as the policy model.
- Loaded a second copy as the frozen reference model.
- Used the `Anthropic/hh-rlhf` preference dataset.
- Prepared tokenizer padding with the EOS token.
- Configured `DPOConfig` with:
  - batch size `1`
  - `beta=0.3`
  - `paged_adamw_8bit`
  - `max_length=512`
- Trained on a small subset of 125 training examples and evaluated on 125 test examples.
- Extracted DPO metrics such as loss, reward margin, reward accuracy, chosen reward, and rejected reward.
- Plotted the logged metrics for analysis.

## Result

The notebook successfully logged DPO-specific training metrics. One recorded early step was:

| Step | Loss | Reward margin | Reward accuracy | Chosen reward | Rejected reward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | `0.706` | `-0.024` | `0.4` | `0.007` | `0.032` |

The reward margin was not yet healthy, and the loss curve had sudden jumps. The notebook notes that `beta=0.3` was probably not well tuned for this small, constrained run.

## How To Run

Open the notebook:

```bash
jupyter notebook Week6_work/DPO_Fine_Tuning.ipynb
```

Run this in Colab or another CUDA environment. DPO needs enough memory to hold the policy model and the reference model.

## Requirements

Relevant Python packages:

- `torch`
- `transformers`
- `datasets`
- `trl`
- `bitsandbytes`
- `accelerate`
- `matplotlib`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA describes the key shift from raw pretrained models to useful assistants. A base LLM is mainly a next-token predictor; instruction tuning, RLHF, and DPO are ways to align that predictor with human intent.

DPO was especially useful to study because it removes the separate reward-model step from classic RLHF. The model directly learns to prefer chosen responses over rejected ones, while the reference model and `beta` control how far the tuned policy is allowed to move from the original behavior.

## Takeaway

DPO was conceptually cleaner than full RLHF, but it was not automatic magic. Preference tuning still depended heavily on prompt format, data size, GPU memory, and the `beta` value controlling how far the policy can move from the reference model.
