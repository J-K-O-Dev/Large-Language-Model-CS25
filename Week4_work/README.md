# Week 4 - Fine-Tuning Pretrained Models

Week 4 moved from building architectures manually to adapting existing pretrained models. The focus was transfer learning: take a model that already learned broad language patterns, then fine-tune it for a specific task.

## Goal

Fine-tune BERT for sentiment classification and GPT-2 for code-style generation.

## Files

| File | Purpose |
| --- | --- |
| `Fine_tune.ipynb` | BERT fine-tuning on SST-2 sentiment classification |
| `GPT2.ipynb` | GPT-2 causal language-model fine-tuning on MBPP coding examples |
| `../Mid-term_Report/assets/week4_bert_trainer_loss.jpg` | Saved BERT training-loss plot |
| `../Mid-term_Report/assets/week4_gpt2_trainer_loss.jpg` | Saved GPT-2 training-loss plot |

## What Was Implemented

- Loaded `bert-base-uncased` with `AutoModelForSequenceClassification`.
- Used the SST-2 sentiment dataset with two labels: positive and negative.
- Tokenized data with the matching BERT tokenizer.
- Used Hugging Face `Trainer`, `TrainingArguments`, and `DataCollatorWithPadding`.
- Loaded GPT-2 with `AutoModelForCausalLM`.
- Formatted MBPP examples as problem statements followed by code solutions.
- Used `DataCollatorForLanguageModeling` with `mlm=False`.
- Saved the fine-tuned model outputs from the notebooks.

## Result

The BERT training curve was noisy at the step level, but its moving average settled around the `0.2-0.3` loss range. The GPT-2 coding fine-tuning run moved from high initial loss to roughly `5` by the end of the logged steps.

The important result was practical: I learned the full Hugging Face fine-tuning loop, including model loading, tokenization, collators, trainer configuration, training, and saving.

## How To Run

Open either notebook:

```bash
jupyter notebook Week4_work/Fine_tune.ipynb
jupyter notebook Week4_work/GPT2.ipynb
```

Colab or another GPU environment is recommended, especially for repeated experiments.

## Requirements

Relevant Python packages:

- `torch`
- `transformers`
- `datasets`
- `accelerate`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA separates the main pretraining paradigms clearly. BERT is encoder-based and learns bidirectional representations through masked language modelling, which makes it strong for understanding tasks like classification. GPT is decoder-only and learns by causal next-token prediction, which makes it natural for generation.

Fine-tuning showed what transfer learning really means in practice. The model already contains broad language knowledge, and the task-specific dataset nudges that knowledge toward sentiment classification or code completion. This also made prompt and dataset formatting feel like part of the model, not just preprocessing.

## Takeaway

Fine-tuning felt easier than writing a Transformer from scratch, but the hard parts did not disappear. They moved into dataset formatting, tokenization, padding, labels, collators, batch size, and evaluation.
