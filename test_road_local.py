from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
import torch
import os

device_id = 0  # Change this to a valid device ID based on the check above
cuda_device = (
    f"cuda:{device_id}"
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id
    else "cpu"
)

# Set environment variable accordingly
if cuda_device.startswith("cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

device = torch.device(cuda_device)
print(f"Using device: {device}")

# Load the fine-tuned model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Johnx69/road-finetuned",  # Change to your trained model
    load_in_4bit=False,  # Set to False for 16bit LoRA
    device_map=cuda_device,
)
FastVisionModel.for_inference(model)


def infer(image_path, instruction):
    """
    Perform inference given an image and an instruction.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    # Construct the message
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    # Tokenize input
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
        # ).to("cuda")
    ).to(device)

    # Stream inference results
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=500,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )


# Example usage
image_path = "pothole_3.png"  # Replace with your image path
instruction = "You are an expert specialized in road assessment and pothole evaluation. Your task is to analyze images of roads, detect potholes, and provide detailed evaluations, including size, depth, severity, and possible safety risks. Generate structured reports with recommendations for repair priority based on road conditions. Answer this question: How significant do you believe the pothole damage is across the evaluated section of the road?"
infer(image_path, instruction)
