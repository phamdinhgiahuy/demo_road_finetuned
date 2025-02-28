import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor


def load_model_and_processor():
    """Load the model and processor for Llama Vision."""
    print("Loading model and processor...")

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor
