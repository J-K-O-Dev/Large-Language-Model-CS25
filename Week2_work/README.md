# Week 2 - LSTM Text Generation

Week 2 moved from fixed-size inputs to sequences. After the NumPy network, I wanted to understand how a model can carry information across time and use previous tokens to predict the next one.

## Goal

Train an LSTM-based text generator and sample text from it after learning patterns in a Shakespeare-style corpus.

## Files

| File | Purpose |
| --- | --- |
| `Copy_of_Character_level_LSTM_model.ipynb` | Main LSTM notebook |
| `../Mid-term_Report/assets/week2_lstm_loss_original.png` | Saved training-loss plot |

## What Was Implemented

- Loaded a text dataset through `kagglehub`.
- Cleaned and tokenized the text into a vocabulary.
- Converted tokens into integer IDs for model input.
- Prepared input-target sequence pairs for next-token prediction.
- Built an LSTM model in PyTorch with an embedding layer, recurrent layers, and a final linear output layer.
- Trained with cross-entropy loss and the Adam optimizer.
- Tracked training loss across epochs.
- Added a generation function that starts from seed text and repeatedly predicts the next token.

## Result

The saved loss curve shows training loss falling from about `10.1` to about `7.2` over 50 epochs. The generated samples were still small and imperfect, but the model moved from random-looking output toward recognizable word and phrase patterns.

## How To Run

Open the notebook:

```bash
jupyter notebook Week2_work/Copy_of_Character_level_LSTM_model.ipynb
```

This notebook is also suitable for Google Colab. A GPU is useful but not strictly required for the small experiment.

## Requirements

Relevant Python packages:

- `torch`
- `matplotlib`
- `kagglehub`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA places this week in the pre-Transformer NLP era: tokenization, embeddings, language modelling, RNNs, and LSTMs. The important idea was that text has order, and a model needs some way to carry information from earlier tokens into later predictions.

LSTMs improve on plain RNNs with gates that decide what to remember, forget, and expose as output. At the same time, this week also showed why Transformers were needed later: recurrent models process tokens sequentially, are harder to parallelize, and still struggle with very long-range dependencies.

## Takeaway

The biggest lesson was that sequence modeling is not just about the model. Data preparation matters a lot: vocabulary, sequence length, input-target alignment, and sampling all affect whether the LSTM learns anything useful.
