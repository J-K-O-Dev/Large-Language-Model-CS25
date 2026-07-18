# Week 9 Report: Exploring Multimodal AI with LLaVA and Stable Diffusion

This week, the project expanded beyond text-only models into the exciting realm of multimodal AI. The focus was on two key capabilities: understanding visual information (image-to-text) and generating visual content from text descriptions (text-to-image). To explore these, I worked with two powerful open-source models: LLaVA and Stable Diffusion, both accessed through the Hugging Face ecosystem.

## 1. Image Understanding with LLaVA (Vision-Language Model)

The first part of this week's work involved using a Vision-Language Model (VLM) to interpret and describe an image.

- **File**: `llava.py`
- **Model**: `llava-hf/llava-v1.6-mistral-7b-hf`, a state-of-the-art VLM that combines a vision encoder with the Mistral-7B language model.
- **Task**: The script loads the LLaVA model and an image (the training graph from the Week 1 report). It then prompts the model to describe what it sees, demonstrating the model's ability to reason about visual data.
- **Technical Details**: To make this large 7-billion-parameter model runnable on consumer hardware, it's loaded using 4-bit quantization via the `BitsAndBytesConfig` from the `transformers` library. This significantly reduces the memory footprint. The script also shows how to construct a multimodal prompt containing both text and image placeholders.

### How to Run `llava.py`

1.  **Install Dependencies**:

    ```bash
    pip install torch transformers bitsandbytes accelerate Pillow
    ```

2.  **Run the Script**:
    ```bash
    python llava.py
    ```

The script will download the model (this may take some time on the first run), load it onto the GPU, and print the model's textual description of the image to the console.

## 2. Image Generation with Stable Diffusion

The second part focused on the creative side of multimodal AI: generating images from text prompts.

- **File**: `difussion.py`
- **Model**: `runwayml/stable-diffusion-v1-5`, a widely-used latent diffusion model.
- **Task**: This script showcases the power of text-to-image generation. It takes a detailed text prompt ("A high-tech research lab...") and uses the Stable Diffusion pipeline from the `diffusers` library to generate a high-quality, photorealistic image.
- **Technical Details**: The model pipeline is loaded in `float16` (half-precision) to optimize for speed and reduce VRAM usage on the GPU. The script runs the inference pipeline for 30 steps and saves the resulting image to a file.

### How to Run `difussion.py`

1.  **Install Dependencies**:

    ```bash
    pip install torch diffusers transformers
    ```

2.  **Run the Script**:
    ```bash
    python difussion.py
    ```

An image file named `generated_lab.png` will be saved in the `Week9_work` directory.

## 3. Learnings and Reflections

This week was a fantastic demonstration of how the transformer architecture has been adapted for more than just text.

- **Complementary Modalities**: It was fascinating to work on both sides of the multimodal coin. LLaVA deconstructs an image into a textual understanding, while Stable Diffusion constructs an image from a textual concept.
- **The Power of the Ecosystem**: Working with these models highlights the power of the Hugging Face ecosystem (`transformers`, `diffusers`). It makes it incredibly straightforward to download, configure, and run state-of-the-art models with just a few lines of code.
- **Hardware and Optimization**: The practical necessity of techniques like 4-bit quantization and half-precision floating points became very clear. These methods are essential for making billion-parameter models accessible without requiring an industrial-scale data center.

Overall, this week successfully bridged the gap from pure language processing to the rich, interconnected world of text and vision.
