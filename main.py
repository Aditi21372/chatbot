from dotenv import load_dotenv
import os

load_dotenv()  # This will read variables from your .env file
my_api_key = os.getenv("OPENAI_API_KEY")
print("API key is:", my_api_key)
