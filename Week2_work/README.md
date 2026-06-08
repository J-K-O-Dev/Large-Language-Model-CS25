### Week 2: My First LSTM for Text Generation

After getting my feet wet with feed-forward networks in Week 1, I was ready to dive into something more complex. This week was all about tackling sequential data with a powerful architecture: the Long Short-Term Memory (LSTM) model. My goal was to build one from the ground up and teach it to generate text, character by character.

#### The Theory Behind LSTMs

Before I could write any code, I had to spend a good chunk of time hitting the books (or, well, the online articles and papers) to really understand what makes LSTMs tick.

- **The Leap from RNNs to LSTMs:** I started with the basics of Recurrent Neural Networks (RNNs). The idea of a hidden state that acts as a 'memory' was cool, but I quickly learned about their major weakness: the vanishing and exploding gradient problems. This makes it tough for them to remember things from way back in a sequence.

- **Inside the LSTM Cell:** That's where the LSTM comes in. It's a clever upgrade to the RNN designed to fix that long-term memory issue. The magic is in its cell structure, which has a few key parts:
  - **Cell State:** Think of it as a conveyor belt for memory, letting information pass through the whole sequence with minimal changes.
  - **Forget Gate:** Decides what old information is no longer relevant and should be thrown away.
  - **Input Gate:** Decides what new information is important enough to be stored.
  - **Output Gate:** Decides what part of the cell's memory to use for the final output at that step.

- **The Math: Forward and Backward:** I then had to trace the path of the data through the network (the forward pass) and, more importantly, the path of the error back through time (Backpropagation Through Time, or BPTT). Getting the math right for how the gradients flow through all those gates was crucial for making the model learn anything.

#### Implementation from Scratch

With the theory down, it was time to get my hands dirty with the implementation.

- **Preparing the Data:** I started with a small text file as my dataset. The first task was to get it ready for the network. This meant creating a vocabulary of every unique character, mapping each one to a number, and then chopping the text up into input sequences and their corresponding target characters.

- **Building the `LSTM` Class:** Just like last week, I built the model inside a Python class to keep things organized. This class was responsible for initializing all the weight and bias matrices for the gates, handling the forward pass to process a sequence, running the backward pass (BPTT) to learn, and updating the parameters. A fun new addition was a `sample` method to see what the model was thinking and generate new text.

- **The Training Loop:** I trained the model over many epochs, feeding it sequences of characters and asking it to predict the next one. The loss function was Cross-Entropy, which makes sense for a classification problem like this (picking the right character out of the whole vocabulary). The best part was periodically sampling text from the model during training. It was amazing to watch it go from spitting out random nonsense to forming actual words and simple sentences.

#### Key Takeaways

This week was a huge leap in complexity from the simple network I built in Week 1. Juggling all the parameters and making sure the gradients were flowing correctly through time (BPTT) took a lot of careful coding and debugging. But, seeing the model actually start to generate coherent text, one character at a time, was an incredibly rewarding "it's alive!" moment.

Building this from scratch really hammered home how RNNs and LSTMs "think" about sequences. It gave me a gut feeling for what's happening under the hood, which I know will be invaluable as we move on to even more advanced architectures. This feels like a critical piece of the puzzle, and I'm excited for what's next.
