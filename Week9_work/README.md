# Week 9 - Multimodal AI With LLaVA And Stable Diffusion

Week 9 moved beyond text-only language models. I explored both sides of multimodal AI: a model that can understand images and a model that can generate images from text.

## Goal

Run image-to-text understanding with LLaVA and text-to-image generation with Stable Diffusion.

## Files

| File | Purpose |
| --- | --- |
| `llava.py` | Loads LLaVA and asks it to describe a training-curve image |
| `difussion.py` | Generates an image from a text prompt with Stable Diffusion |
| `week1_loss_accuracy.jpg` | Image input used by the LLaVA script |
| `generated_lab.png` | Stable Diffusion output image |

## What Was Implemented

### LLaVA Image Understanding

- Loaded `llava-hf/llava-v1.6-mistral-7b-hf`.
- Used `LlavaNextProcessor`, tokenizer, and image processor.
- Loaded the model with 4-bit quantization through `BitsAndBytesConfig`.
- Opened `week1_loss_accuracy.jpg` with Pillow.
- Built a prompt containing the `<image>` placeholder.
- Generated a textual description of the image.

### Stable Diffusion Image Generation

- Loaded `runwayml/stable-diffusion-v1-5`.
- Used `StableDiffusionPipeline` from `diffusers`.
- Loaded the pipeline in `torch.float16`.
- Generated an image from the prompt:

```text
A high-tech research lab with glowing holographic matrices, cinematic lighting, photorealistic, 8k resolution
```

- Saved the result as `generated_lab.png`.

## Result

The LLaVA script produced a text description of the Week 1 training plot, showing how a vision-language model can interpret a chart-like image.

The Stable Diffusion script generated and saved a photorealistic research-lab image at:

```text
Week9_work/generated_lab.png
```

## How To Run

Run from inside `Week9_work`, because both scripts use relative file paths:

```bash
cd Week9_work
python llava.py
python difussion.py
```

A CUDA GPU is expected. Both scripts call `.to("cuda")`, and LLaVA is a large model even with 4-bit quantization.

## Requirements

Relevant Python packages:

- `torch`
- `transformers`
- `bitsandbytes`
- `accelerate`
- `pillow`
- `diffusers`

They are included in the root `requirements.txt`.

## Learning From The PoA

The PoA expands the Transformer story beyond text. Vision Transformers treat images as patches, CLIP connects images and text with contrastive learning, and vision-language models such as LLaVA attach visual understanding to an LLM so it can answer questions about images.

Stable Diffusion represents the generative side of vision. Instead of classifying or describing an image, it learns to reverse a noise process and create an image from a text prompt. Running both LLaVA and Stable Diffusion made the difference clear: one model reads visual information into language, while the other turns language into visual content.

## Takeaway

This week showed that the same ecosystem used for language models can extend naturally into vision. LLaVA turns an image into language-level reasoning, while Stable Diffusion turns language into an image. Together, they make the project feel less like "only NLP" and more like a doorway into general multimodal systems.
