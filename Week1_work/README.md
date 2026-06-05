### Week 1 Report: Building a Neural Network from Scratch

This week marked the beginning of my journey into building Large Language Models from scratch, as part of the project with the Mathematics and Physics club at IIT Bombay. The primary focus was on understanding the fundamental building blocks of neural networks, both theoretically and practically.

#### Mathematical Foundations

A significant portion of the week was dedicated to diving into the mathematics that powers neural networks. I started with the basics of a simple two-layer neural network architecture. Key concepts I explored include:

- **Forward Propagation:** Understanding how input data flows through the network, gets transformed by weights, biases, and activation functions to produce an output. I implemented this using the sigmoid activation function.
- **Loss Function:** To measure how well our model is performing, I learned about and implemented the Binary Cross-Entropy loss function, which is suitable for the binary classification task I was tackling.
- **Backpropagation and Gradient Descent:** This was the most challenging yet rewarding part. I worked through the calculus to understand how the error is propagated backward through the network. This allowed me to calculate the gradients of the loss function with respect to the weights and biases, which are then used to update the model's parameters via gradient descent.

#### Implementation from Scratch

Putting theory into practice, I built a neural network from the ground up using Python and NumPy. This hands-on experience was invaluable for solidifying my understanding.

- **The `Model` Class:** I created a Python class to encapsulate the neural network. It handles parameter initialization, forward pass (prediction), backward pass (learning), and the main training loop.
- **Dataset:** I used the `make_moons` dataset from `scikit-learn` to create a non-linearly separable binary classification problem, which is a classic test for a simple neural network.
- **Training and Evaluation:** The model was trained over a number of epochs. During training, I monitored the loss and accuracy to see the learning progress. I also implemented a mechanism to save the trained model using `pickle`.
- **Testing:** A separate script (`Testing.py`) was written to load the saved model and test its performance on a new set of data, confirming that the model had learned to generalize.

#### Key Takeaways

This week was a deep dive into the core mechanics of neural networks. While libraries like PyTorch or TensorFlow abstract away much of this complexity, building it from scratch with NumPy provided a crucial intuition for what's happening under the hood. This foundational knowledge feels essential as we move towards more complex architectures in the coming weeks. The successful implementation and training of this simple model have been a great confidence booster.
