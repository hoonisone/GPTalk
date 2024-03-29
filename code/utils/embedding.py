import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
import openai

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name):
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    def __call__(self, input: Documents) -> Embeddings:
        return self.embedder(input)

class OpenAIEmbeddingFunction():
    def __init__(self):
        self.client = openai.OpenAI()
        
    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(input=input, model='text-embedding-ada-002')
        return response.data[0].embedding

class OpenAIEmbeddingFunction2():
    def __init__(self):
        self.client = openai.OpenAI()
        
    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(input=input, model='text-embedding-ada-002')
        return [response.data[0].embedding]

def mldb_default_ebmedding():
    return OpenAIEmbeddingFunction()

def get_default_embedding():
    return OpenAIEmbeddingFunction2()
    return MyEmbeddingFunction("all-MiniLM-L6-v2")
