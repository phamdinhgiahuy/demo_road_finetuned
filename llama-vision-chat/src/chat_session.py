import torch
from PIL import Image
import os


class ChatSession:
    def __init__(self, model, processor):
        """Initialize a chat session with the given model and processor."""
        self.model = model
        self.processor = processor
        self.messages = []

    def process_input(self, user_input):
        """Process user input and return the model's response."""
        if user_input.lower().startswith("image:"):
            # Handle image input
            image_path = user_input[6:].strip()
            return self.process_image(image_path)
        else:
            # Handle text-only input
            return self.process_text(user_input)

    def process_image(self, image_path):
        """Process an image and the associated query."""
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"

        try:
            image = Image.open(image_path)

            # Add image message to the conversation
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": "What is showing in this image? Please describe it.",
                        },
                    ],
                }
            )

            # Process the input
            input_text = self.processor.apply_chat_template(
                self.messages, add_generation_prompt=True
            )
            inputs = self.processor(
                image, input_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)

            # Generate response
            output = self.model.generate(**inputs, max_new_tokens=256)
            response = self.processor.decode(output[0], skip_special_tokens=True)

            # Add assistant response to the conversation history
            self.messages.append({"role": "assistant", "content": response})

            return response
        except Exception as e:
            return f"Error processing the image: {str(e)}"

    def process_text(self, text):
        """Process a text-only query."""
        try:
            # Check if there's an image in conversation history
            has_image = any(
                msg.get("role") == "user"
                and any(
                    content.get("type") == "image"
                    for content in msg.get("content", [])
                    if isinstance(content, dict)
                )
                for msg in self.messages
            )

            # If we have previous images in the conversation and the model requires images
            if has_image:
                # We need to get the most recent image from conversation history
                for msg in reversed(self.messages):
                    if msg.get("role") == "user":
                        for content in msg.get("content", []):
                            if (
                                isinstance(content, dict)
                                and content.get("type") == "image"
                            ):
                                image = content.get("image")
                                if image:
                                    # Add text message with the existing image
                                    self.messages.append(
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "image": image},
                                                {"type": "text", "text": text},
                                            ],
                                        }
                                    )

                                    # Process the input with image
                                    input_text = self.processor.apply_chat_template(
                                        self.messages, add_generation_prompt=True
                                    )
                                    inputs = self.processor(
                                        image,
                                        input_text,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    ).to(self.model.device)

                                    # Generate response
                                    output = self.model.generate(
                                        **inputs, max_new_tokens=256
                                    )
                                    response = self.processor.decode(
                                        output[0], skip_special_tokens=True
                                    )

                                    # Add assistant response to the conversation history
                                    self.messages.append(
                                        {"role": "assistant", "content": response}
                                    )

                                    return response
                                break

                return "Error: Previous image could not be retrieved from conversation history."
            else:
                # Regular text-only processing for conversations without images
                self.messages.append(
                    {"role": "user", "content": [{"type": "text", "text": text}]}
                )

                # Process the input
                input_text = self.processor.apply_chat_template(
                    self.messages, add_generation_prompt=True
                )
                inputs = self.processor(text=input_text, return_tensors="pt").to(
                    self.model.device
                )

                # Generate response
                output = self.model.generate(**inputs, max_new_tokens=256)
                response = self.processor.decode(output[0], skip_special_tokens=True)

                # Add assistant response to the conversation history
                self.messages.append({"role": "assistant", "content": response})

                return response
        except Exception as e:
            return f"Error processing the text: {str(e)}"
