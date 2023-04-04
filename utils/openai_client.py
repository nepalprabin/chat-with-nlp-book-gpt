import os
from langchain.embeddings.openai import OpenAIEmbeddings


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def embeddings_(openai_api_key):
    return OpenAIEmbeddings(openai_api_key=openai_api_key)
