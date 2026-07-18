import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer, LlavaNextImageProcessor
from PIL import Image


quantization=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id="llava-hf/llava-v1.6-mistral-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
image_processor = LlavaNextImageProcessor.from_pretrained(model_id)
processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)

model=LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization,
    device_map='auto'
)

image=Image.open("week1_loss_accuracy.jpg").convert("RGB")

prompt="[INST] <image>\nDescribe what you see in this image and explain the humor or context if any. [/INST]"

input=processor(prompt=prompt,images=image,return_tensors="pt").to("cuda")
with torch.inference_mode():
    output=model.generate(
        **input,
        max_new_tokens=200,
        do_sample=False
    )

# Corrected skip_special_token to skip_special_tokens
response=processor.decode(output[0],skip_special_tokens=True)

print("\n--- Model Response ---")
print(response.split("[INST]")[-1].strip())