import os
from dotenv import load_dotenv
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from haystack.components.generators.utils import print_streaming_chunk
def simple_print_callback(chunk):
    print(chunk)

try:
    # Set your API key as environment variable before executing this
    load_dotenv()
    API_KEY = os.environ.get('API_KEY')
    print(f"API_KEY is set: {API_KEY is not None}")  # Checkpoint 1

    chat_generator = OpenAIChatGenerator(api_key=Secret.from_env_var("API_KEY"),
        api_base_url="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
        streaming_callback=simple_print_callback)  # Using a simpler callback for testing
    chat_generator.run(messages=[ChatMessage.from_user("Return this text: 'test'")])

    print("Script execution completed.")  # Checkpoint 2
except Exception as e:
    print(f"An error occurred: {e}")