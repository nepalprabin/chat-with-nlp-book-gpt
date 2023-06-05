import os
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# loading environment variables
load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def embeddings_(openai_api_key):
    return OpenAIEmbeddings(openai_api_key=openai_api_key)
