import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return [0]
    
def get_client(mode:str = None):
    
    if mode == "http_client":
        server_ip = os.getenv("CHROMA_SERVER_IP")
        server_port = os.getenv("CHROMA_SERVER_PORT")
        return chromadb.HttpClient(host=server_ip, port=server_port)
    else:
        return chromadb.Client()

def get_embedding_function(mode:str = None):
    if mode == "my":
        return MyEmbeddingFunction()
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")