import os
import lmdb
import json
import openai
import numpy as np
from utils.simulatiry import CosinSimilarity
from utils.embedding import mldb_default_ebmedding

client = openai.OpenAI()


    
