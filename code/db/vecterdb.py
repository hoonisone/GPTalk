import os
import lmdb
import json
import openai
import numpy as np

client = openai.OpenAI()

class VecterDB:
    DB_DIR = os.path.dirname(__file__)
    
    def __init__(self):
        self.env = lmdb.open(VecterDB.DB_DIR, map_size=10485760*4)
    
    @staticmethod
    def similarity(v1, v2):  # return dot product of two vectors
        return np.dot(v1, v2)
    
    @staticmethod
    def get_embedding(content, model='text-embedding-ada-002'):
        response = client.embeddings.create(input=content, model=model)
        vector = response.data[0].embedding
        return vector
    
    def make_vector_db(self, df):
        with self.env.begin(write=True) as txn:
            for id, [question, answer] in df[["question", "answer"]].iterrows():
                txn.put(f"Q_{id}".encode(), question.encode())
                txn.put(f"A_{id}".encode(), answer.encode())
                
                question = json.dumps(VecterDB.get_embedding(question))
                answer = json.dumps(VecterDB.get_embedding(answer))
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
        QE = self.get_embedding(question)
        answers = []
        for id in range(num):
            qe = self.get_qe(id)
            sim = self.similarity(QE, qe)
            answers.append([sim, id])
            answers.sort(reverse=True)
            if len(answers) > t_num:
                answers.pop()
        
        return [{"question": self.get_q(id), "answer": self.get_a(id), "similarity": sim} for sim, id in answers]
    
