### Week 3 Report: Taming the Transformer

This week was a quantum leap. After building my own LSTM, I felt like I had a decent grasp on sequence modeling. Then came the Transformer. It felt like graduating from building go-karts to assembling a jet engine. The goal was to implement a decoder-only Transformer, the very architecture that powers models like GPT, and honestly, I was both thrilled and a little terrified.

#### The Theory: "Attention is All You Need"

The jump from RNNs/LSTMs to Transformers is a complete paradigm shift. Instead of processing tokens one by one and trying to cram everything into a hidden state, the Transformer can look at the entire sequence at once. The secret sauce is **self-attention**.

- **Self-Attention:** This mechanism allows every token in the input to look at every other token and decide which ones are most important for understanding its own meaning. It's like being in a conversation where you can instantly weigh the importance of every word someone has said, no matter how long ago they said it. This is what makes Transformers so powerful at capturing long-range dependencies and what allows them to be massively parallelized on GPUs.

- **Modern Enhancements:** I didn't just build a vanilla Transformer. I dove into some of the newer, more efficient components that are used in state-of-the-art models:
  - **Grouped-Query Attention (GQA):** A smart optimization over standard Multi-Head Attention. Instead of every query head having its own key and value, several query heads share them. This drastically cuts down on the memory and computation needed, especially during generation.
  - **Rotary Positional Embeddings (RoPE):** This was mind-bending. Instead of just adding positional information to the embeddings, RoPE _rotates_ the query and key vectors based on their position. It's a more elegant way to encode relative positional information, which has proven to be incredibly effective.
  - **SwiGLU Feed-Forward:** The feed-forward layers got an upgrade too. I implemented SwiGLU, a variant of Gated Linear Units, which acts as a more effective activation function than a simple ReLU.
  - **RMSNorm:** Instead of the standard Layer Normalization, I used RMSNorm. It's simpler, faster, and works just as well, if not better.

#### Implementation: Standing on the Shoulders of Giants (PyTorch)

Building this from scratch with NumPy would have been a Herculean task. This week, I made the move to **PyTorch**, and it was a game-changer.

- **The `transformer_model.py`:** This file is the heart of the project. I structured the model into logical `nn.Module` classes:
  - The `Block` class, which is the core repeating unit, containing a `GroupQueryAttentionHead` and a `SwiGLUFeedForward` network, with RMSNorm applied before each.
  - The main `MY_Gpt` class, which stacks these blocks, adds the token and positional embeddings, and tops it off with a final layer to predict the next token.
- **Tokenizer:** I graduated from a simple character-level vocabulary to `tiktoken`, the same tokenizer used by GPT-2. This gives the model a much richer understanding of language from the get-go, with a vocabulary of over 50,000 tokens.
- **Training:** The training loop was pretty standard PyTorch fare: get a batch, do a forward pass, calculate the loss (cross-entropy), backpropagate, and update the weights with the AdamW optimizer. Seeing the validation loss drop was a huge relief.
- **Generation:** The `generate` method is where the magic happens. Watching the model spit out coherent, Shakespeare-esque text after training was just... wow. It felt like I had breathed life into the machine.

#### Key Takeaways

This week was an absolute whirlwind. The complexity was an order of magnitude higher than the LSTM, and there were moments I felt completely lost in the math and code. But seeing it all come together... it's a feeling I won't forget.

The move to PyTorch felt like taking the training wheels off. It handles all the gradient calculations, letting me focus on the high-level architecture. Implementing things like RoPE and GQA, which are genuinely modern techniques, made me feel like I was working on the cutting edge.

The difference in output quality between this and last week's LSTM is night and day. The Transformer doesn't just mimic patterns; it seems to have a nascent understanding of structure and context. It's still just predicting the next token, but the way it does it is so much more sophisticated. This week, more than any other, made me truly appreciate the "magic" of modern LLMs. It's not magic, it's just incredible engineering, and I got to build a piece of it myself. I'm buzzing with excitement for what's next.
