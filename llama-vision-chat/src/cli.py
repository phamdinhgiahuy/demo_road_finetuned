import sys
from chat_session import ChatSession
from model_loader import load_model_and_processor


def main():
    print("Welcome to the Llama Vision Chat!")
    print("Type 'exit' to end the session.")
    print("Loading model, please wait...")

    # Load the model and processor
    model, processor = load_model_and_processor()

    # Initialize the chat session with the model and processor
    chat_session = ChatSession(model, processor)

    print("Model loaded! You can now chat.")
    print("To add an image, type 'image: [path_to_image]'")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending the session. Goodbye!")
            break

        response = chat_session.process_input(user_input)
        print(f"Model: {response}")


if __name__ == "__main__":
    main()
