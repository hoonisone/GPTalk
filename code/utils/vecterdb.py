import os
import lmdb
import json
import chromadb
from abc import ABC, abstractmethod
from overrides import overrides
from utils.embedding import get_default_embedding, mldb_default_ebmedding
from utils.simulatiry import CosinSimilarity
from typing import List, Callable

class VectorDB(ABC):
    @abstractmethod
    def __init__(self, collection_name:str, embedding_function:Callable=None):
        pass
            
    @abstractmethod
    def add(self, id:str, document:str, metadata:dict=None)->None:
        pass
    
    @abstractmethod
    def query(self, query_text:List[str], n_results:int=1):
        pass
    
    @abstractmethod
    def count(self):
        pass
    
    @abstractmethod
    def clear(self)->None:
        pass    
        
class ChromaDB(VectorDB):
    @overrides
    def __init__(self, collection_name:str, embedding_function:Callable=None):
        embedding_function = embedding_function if embedding_function != None else get_default_embedding()
        self.collection_name = collection_name
        
        self.client = self.get_client()
        self._collection = None
    
    @property
    def collection(self):
        if self._collection == None:
            self._collection = self.client.get_or_create_collection(name=self.collection_name)
        return self._collection
    @staticmethod
    def get_client():
        server_ip = os.getenv("CHROMA_SERVER_IP")
        server_port = os.getenv("CHROMA_SERVER_PORT")
        return chromadb.HttpClient(host=server_ip, port=server_port)
    
    @overrides
    def add(self, id:str, document:str, metadata:dict=None)->None:
        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[id]
        )
    
    @overrides
    def query(self, query_text:List[str], n_results:int=1):
        return self.collection.query(
            query_texts=query_text,
            n_results=n_results
        )
        
    @overrides
    def count(self):
        return self.collection.count()
    
    @overrides
    def clear(self)->None:
        self.client.delete_collection(name=self.collection_name)
        self._collection = None
    
class LMDBVecterDB:
    DB_DIR = os.path.dirname(__file__)
    
    def __init__(self):
        self.env = lmdb.open(LMDBVecterDB.DB_DIR, map_size=10485760*4*1024)
        self.similarity = CosinSimilarity()
        self.embedding = mldb_default_ebmedding()
    
    def make_vector_db(self, df):
        with self.env.begin(write=True) as txn:
            for id, [question, answer] in df[["question", "answer"]].iterrows():
                txn.put(f"Q_{id}".encode(), question.encode())
                txn.put(f"A_{id}".encode(), answer.encode())
                
                question = json.dumps(self.embedding(question))
                answer = json.dumps(self.embedding(answer))
                txn.put(f"QE_{id}".encode(), question.encode())
                txn.put(f"AE_{id}".encode(), answer.encode())
            
            txn.put(b"num_samples", str(len(df)).encode())
                
    def get_num(self):
        with self.env.begin() as txn:
            return int(txn.get(b'num_samples').decode()) 
            
    def get_qe(self, idx):
        with self.env.begin() as txn:
            question = json.loads(txn.get(f"QE_{idx}".encode()).decode())
            return question
        
    def get_ae(self, idx):
        with self.env.begin() as txn:
            question = json.loads(txn.get(f"AE_{idx}".encode()).decode())
            return question
        
    def get_q(self, idx):
        with self.env.begin() as txn:
            question = txn.get(f"Q_{idx}".encode()).decode()
            return question
        
    def get_a(self, idx):
        with self.env.begin() as txn:
            question = txn.get(f"A_{idx}".encode()).decode()
            return question
    
    def __del__(self):
        self.env.close()

    def get_similer_history(self, question, t_num):
        num = self.get_num()
        QE = self.embedding(question)
        answers = []
        for id in range(num):
            qe = self.get_qe(id)
            sim = self.similarity(QE, qe)
            answers.append([sim, id])
            answers.sort(reverse=True)
            if len(answers) > t_num:
                answers.pop()
        
        return [{"question": self.get_q(id), "answer": self.get_a(id), "similarity": sim} for sim, id in answers]