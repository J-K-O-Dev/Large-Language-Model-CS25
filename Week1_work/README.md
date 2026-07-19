# Week 1 - Neural Network From Scratch

Week 1 was about opening the black box. Before moving toward language models, I built a small neural network with only NumPy so that the basic learning loop was completely visible: predictions, loss, gradients, parameter updates, and evaluation.

## Goal

Build and test a two-layer neural network for binary classification on a non-linear toy dataset.

## Files

| File | Purpose |
| --- | --- |
| `Week1_perceptron.py` | Main NumPy implementation, training loop, save/load logic |
| `Testing.py` | Loads the saved model and checks accuracy on fresh `make_moons` data |
| `../Mid-term_Report/assets/week1_loss_accuracy.jpg` | Saved training plot used in the report |

## What Was Implemented

- Generated a `make_moons` dataset with scikit-learn.
- Built a two-layer neural network with one hidden layer and one sigmoid output.
- Implemented the forward pass manually with NumPy matrix operations.
- Used binary cross-entropy as the loss function.
- Derived and coded backpropagation by hand.
- Updated weights and biases using gradient descent.
- Added accuracy calculation with a `0.5` decision threshold.
- Saved and loaded the trained model using `pickle`.
- Wrote a separate testing script to verify the saved model.

## Result

The model learned the non-linear decision boundary clearly. From the saved plot, binary cross-entropy dropped from about `0.69` to about `0.26`, while accuracy improved to roughly `0.88-0.89`.

That result was enough to confirm that the hand-written forward pass, backward pass, and update rules were working.

## How To Run

From the repository root:

```bash
python Week1_work/Week1_perceptron.py
python Week1_work/Testing.py
```

The first command trains the model if `first_model.pkl` is not already present. The second command loads the saved model and prints predictions plus final accuracy.

## Requirements

Relevant Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

They are included in the root `requirements.txt`.

## Learning From The PoA

The Plan of Action frames this week as the foundation week: linear algebra, calculus, probability, and Python fluency. In this implementation, matrix multiplication became the forward pass, the chain rule became backpropagation, and binary cross-entropy connected the model directly to probability.

The main learning was that neural networks are not mysterious at the small scale. They are repeated numerical steps: initialize weights, predict, measure error, compute gradients, update, and repeat. Writing the MLP without PyTorch autograd made every shape and derivative visible.

## Takeaway

This week made backpropagation feel real. Libraries are helpful later, but writing the gradients by hand made it much easier to understand what a neural network is actually doing when it "learns."
