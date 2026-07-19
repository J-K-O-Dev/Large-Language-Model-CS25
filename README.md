# Large Language Model CS25

This repository is my week-by-week journey through the core ideas behind modern Large Language Models. The project starts with a tiny neural network written from scratch, then climbs through LSTMs, Transformers, fine-tuning, evaluation, alignment, quantization, RAG, and finally multimodal models.

The aim was not just to call a ready-made API. I wanted to understand the stack from the inside: how gradients update weights, how sequence models remember context, why attention changed NLP, how pretrained models are adapted, and what engineering tricks make large models usable on limited hardware.

Project: Summer of Science 2025, Mathematics and Physics Club, IIT Bombay  
Mentee: Jithin K  
Mentor: Sayandeep Haldar

## Repository Map

| Path | What it contains |
| --- | --- |
| `Week1_work/` | NumPy neural network, manual backpropagation, testing script |
| `Week2_work/` | Character/word-level LSTM text generation notebook |
| `Week3_work/` | Decoder-only Transformer implementation in PyTorch |
| `Week4_work/` | BERT sentiment fine-tuning and GPT-2 code fine-tuning notebooks |
| `Week5_work/` | GPT-2 scaling/perplexity study and TinyLlama loading experiment |
| `Week6_work/` | Direct Preference Optimization with TinyLlama |
| `Week7_work/` | 4-bit quantization, PEFT LoRA adapters, and QLoRA-style DPO setup |
| `Week8_work/` | RAG pipeline, ChromaDB vector store, Flask backend, React chat UI |
| `Week9_work/` | LLaVA image understanding and Stable Diffusion image generation |
| `Mid-term_Report/` | Written mid-term report and result plots for Weeks 1-4 |
| `requirements.txt` | Python dependencies used across scripts and notebooks |

## Weekly Summary And Results

| Week | Focus | Main work | Result |
| --- | --- | --- | --- |
| 1 | Neural network from scratch | Built a two-layer NumPy classifier for `make_moons`, with manual forward pass, binary cross-entropy, backpropagation, gradient descent, saving, and loading. | Loss fell from about `0.69` to `0.26`; accuracy improved to about `0.88-0.89`. |
| 2 | LSTM text generation | Trained an LSTM text generator on Shakespeare-style text, including vocabulary creation, sequence batching, embedding, recurrent layers, and sampling. | Training loss dropped from about `10.1` to `7.2` over 50 epochs, and the model began producing recognizable text patterns. |
| 3 | Decoder-only Transformer | Implemented a GPT-style model with `tiktoken`, causal self-attention, grouped-query attention, RoPE, RMSNorm, SwiGLU, AdamW training, and sampling. | Train/validation cross-entropy dropped from about `10.9` to `6.7`; the trained model generated sample text. |
| 4 | Fine-tuning pretrained models | Fine-tuned `bert-base-uncased` on SST-2 sentiment classification and GPT-2 on MBPP coding examples using Hugging Face `Trainer`. | BERT training loss stabilized around the `0.2-0.3` range; GPT-2 coding loss moved down to about `5`. |
| 5 | Evaluation and model loading | Compared GPT-2 variants on WikiText-2 perplexity and loaded TinyLlama for a practical chat-model inference experiment. | Perplexity improved with scale: roughly `19.6` for GPT-2 124M, `15.3` for 355M, `13.4` for 774M, and `12.6` for 1.5B. |
| 6 | Preference alignment | Used DPO with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on `Anthropic/hh-rlhf`, with a frozen reference model and preference pairs. | DPO metrics were logged; the first logged step showed loss `0.706`, margin `-0.024`, and accuracy `0.4`, revealing that beta and data size needed tuning. |
| 7 | QLoRA-style efficient tuning | Loaded `Meta-Llama-3-8B-Instruct` in 4-bit NF4, added LoRA adapters, and trained with DPO-style preference tuning on a small `hh-rlhf` subset. | Only `41,943,040` of `8,072,204,288` parameters were trainable (`0.5196%`); the recorded train loss was about `0.635`. |
| 8 | RAG chatbot | Built document loading, chunking, embeddings, ChromaDB storage, retrieval, TinyLlama generation, a Flask `/chat` API, and a React chat interface. | Loaded 227 PDF document pages from six LLM papers and built a local retrieval-backed chatbot with persistent frontend history. |
| 9 | Multimodal AI | Ran LLaVA for image understanding and Stable Diffusion for text-to-image generation. | LLaVA described a training plot, and Stable Diffusion saved `Week9_work/generated_lab.png` from a research-lab prompt. |

## Setup

Use Python 3.10+ if possible. A CUDA GPU is strongly recommended for the larger model, RAG, quantization, and multimodal work.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Some notebooks use gated Hugging Face models such as Llama 3. For those, log in first and make sure your account has accepted the model license:

```bash
huggingface-cli login
```

For Week 8's React frontend, Node.js and npm are also required. The frontend dependencies live in `Week8_work/react_interface/package.json`.

## How To Run The Main Pieces

Week 1 NumPy model:

```bash
python Week1_work/Week1_perceptron.py
python Week1_work/Testing.py
```

Week 3 Transformer:

```bash
python Week3_work/transformer_model.py
```

Most other model training work is in notebooks and is best run in Jupyter or Google Colab, especially where GPU memory is needed:

```bash
jupyter notebook
```

Week 8 RAG backend:

```bash
cd Week8_work\notebook
python app.py
```

Week 8 React frontend, in a second terminal:

```bash
cd Week8_work\react_interface
npm install
npm start
```

The backend runs on `http://localhost:5000`, and the React app usually opens on `http://localhost:3000`.

Week 9 multimodal scripts:

```bash
cd Week9_work
python llava.py
python difussion.py
```

## Data And Models Used

The project uses a mix of small local datasets, public Hugging Face datasets, and downloaded model checkpoints:

- `make_moons` from scikit-learn for Week 1.
- Tiny Shakespeare-style text for Week 2.
- `Week3_work/input.txt` with GPT-2 tokenization for Week 3.
- `stanfordnlp/sst2` and `google-research-datasets/mbpp` for Week 4.
- WikiText-2 test text for Week 5 perplexity evaluation.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for Week 5 loading, Week 6 DPO, and Week 8 RAG generation.
- `Anthropic/hh-rlhf` for Weeks 6 and 7 preference tuning.
- Six LLM research PDFs in `Week8_work/data/` for RAG.
- `llava-hf/llava-v1.6-mistral-7b-hf` and `runwayml/stable-diffusion-v1-5` for Week 9.

## Practical Notes

Large model runs can be slow or fail on CPU-only machines. The Llama, LLaVA, Stable Diffusion, DPO, and QLoRA-style experiments were designed with GPU/Colab-style hardware in mind.

`bitsandbytes` and 4-bit quantization are especially hardware-sensitive. If they fail on a local Windows setup, running the notebook in Colab with a CUDA GPU is usually the simplest path.

The Week 8 vector store is persistent. On the first backend run, documents are embedded and saved under `Week8_work/data/vector_store/`; later runs can reuse that store.

## Learning Roadmap

The learning notes in each weekly README are based on the project Plan of Action document: `https://docs.google.com/document/d/1ypGpqkE5_6IT7tHIDRMW92f2qZQixFfV/edit`.

The repo currently contains Weeks 1-9. The PoA also includes a Week 10 direction: frontier models, evaluation, safety, and research thinking. A natural next step for this project would be to benchmark a few open-source models on tasks such as GSM8K, TruthfulQA, HumanEval, or MMLU using an evaluation harness, then write a short research note on capability, hallucination, bias, and prompt-injection risks.

## Final Reflection

The project ended up being a ladder. Week 1 made learning visible at the level of gradients. Week 2 showed why sequence memory matters. Week 3 made the Transformer feel concrete instead of mysterious. Weeks 4 and 5 moved into the practical world of pretrained models, evaluation, and scale. Weeks 6 and 7 explored alignment and efficient adaptation. Week 8 turned the ideas into a usable RAG application. Week 9 opened the door from language-only systems into multimodal AI.

The biggest takeaway is that LLMs are not one trick. They are a careful stack of math, data, architecture, optimization, tooling, hardware choices, and evaluation. Building each piece, even in a small way, made the full system much easier to understand.
