import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

image_path = "pothole.jpg"
image = Image.open(image_path)


# model_id = "Johnx69/road-finetuned"  llava-hf/llava-1.5-7b-hf
model_id = "llava-hf/llava-1.5-7b-hf"

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "What is showing in the image? Please describe in detail.",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=256)
res = processor.batch_decode(generate_ids, skip_special_tokens=True)

# show the result
print(res)
